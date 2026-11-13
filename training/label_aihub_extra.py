#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""label_aihub.py + AI Hub 폴리곤 밖에 블록을 추가하는 기능.

label_aihub.py(팀원 작성)를 고치지 않고 따로 만든 버전이다.
입력·출력 형식은 같다 (--input 에 instances_cvat.json + images_for_cvat/).

키
  1 / 2       주황 영역 전체를 선형 / 점형으로 저장
  3           혼합: 주황 영역 안의 점형 부분을 클릭 → Enter, 마지막에 S
  L / D       주황 영역 밖에 빠진 선형 / 점형 블록 추가: 클릭 → Enter (여러 개 가능)
  Enter       그리던 다각형 완성
  Backspace, 오른쪽 클릭   그리던 점 하나 지우기
  Z           마지막으로 완성한 다각형 지우기
  X 제외, ←/→ 이동, E 내보내기, Q 종료

순서: 빠진 블록을 L/D로 먼저 추가한 다음, 주황 영역을 1/2/3으로 정하면 저장된다.
추가한 블록이 주황 영역과 겹치면 겹친 부분은 추가한 블록의 클래스를 따른다.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageTk
from shapely.geometry import GeometryCollection, box
from shapely.ops import unary_union

from label_aihub import labels_for, polygons_of, safe_polygon, union_coco_polygons

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    tk = None

EMPTY = GeometryCollection()
COLORS = {  # (채우기, 테두리)
    "dot": ("#7449C8", "#BCA8FF"),
    "extra_linear": ("#1E9E8F", "#7FFFE8"),
    "extra_dot": ("#C8497A", "#FFA8D0"),
}
KIND_NAME = {"dot": "혼합 점형", "extra_linear": "추가 선형(L)", "extra_dot": "추가 점형(D)"}


def _norm_poly(points, w, h):
    return safe_polygon([(x * w, y * h) for x, y in points])


def _union(polys):
    polys = [p for p in polys if p is not None]
    return unary_union([q for p in polys for q in polygons_of(p)]) if polys else EMPTY


def build_classes(base, decision, width, height):
    """주황 영역(base) 결정 + 추가 블록 → {'linear': geom, 'dot': geom}."""
    mode = decision["mode"]
    if mode == "skip":
        return {"linear": EMPTY, "dot": EMPTY}

    if mode == "linear":
        linear, dot = base, EMPTY
    elif mode == "dot":
        linear, dot = EMPTY, base
    elif mode == "mixed":
        marked = _union(_norm_poly(p, width, height) for p in decision.get("dots", []))
        if marked.is_empty:
            raise ValueError("혼합인데 점형 영역이 없음")
        dot = base.intersection(marked)
        linear = base.difference(dot)
        if marked.area and dot.area / marked.area < 0.20:
            raise ValueError("점형 영역이 거의 주황 영역 밖에 있음. 밖의 블록은 L/D로 추가하세요")
    else:
        raise ValueError("Unknown mode: " + str(mode))

    frame = box(0, 0, width, height)
    ex = decision.get("extra", [])
    ex_lin = _union(_norm_poly(e["pts"], width, height) for e in ex if e["cls"] == "linear").intersection(frame)
    ex_dot = _union(_norm_poly(e["pts"], width, height) for e in ex if e["cls"] == "dot").intersection(frame)
    ex_dot = ex_dot.difference(ex_lin)  # 추가 선형·점형이 겹치면 선형 우선
    ex_all = unary_union([g for g in (ex_lin, ex_dot) if not g.is_empty]) if (ex_lin.area or ex_dot.area) else EMPTY

    if not ex_all.is_empty:
        linear = linear.difference(ex_all)
        dot = dot.difference(ex_all)
    linear = unary_union([g for g in (linear, ex_lin) if not g.is_empty]) if (linear.area or ex_lin.area) else EMPTY
    dot = unary_union([g for g in (dot, ex_dot) if not g.is_empty]) if (dot.area or ex_dot.area) else EMPTY

    if mode == "mixed" and (dot.area < 100 or linear.area < 100):
        raise ValueError("한 클래스 면적이 너무 작음: 점형 외곽을 확인하세요")
    for p in polygons_of(linear) + polygons_of(dot):
        if p.interiors:
            raise ValueError(
                "구멍이 생김: 점형/추가 영역이 다른 영역 한가운데 떠 있어요.\n"
                "다각형을 블록 줄 끝에서 끝까지 가로지르게(주황 선 바깥까지) 다시 그리세요."
            )
    return {"linear": linear, "dot": dot}


class LabelApp:
    def __init__(self, root, source: Path, dest: Path):
        self.root, self.source, self.dest = root, source, dest
        self.data = json.loads((source / "instances_cvat.json").read_text(encoding="utf-8-sig"))
        self.byid = defaultdict(list)
        for a in self.data.get("annotations", []):
            self.byid[a["image_id"]].append(a)
        self.entries = [i for i in self.data.get("images", []) if (source / "images_for_cvat" / i["file_name"]).is_file()]
        if not self.entries:
            raise RuntimeError("No images found; check input folder and images_for_cvat")
        dest.mkdir(parents=True, exist_ok=True)
        self.progress_file = dest / "quick_label_progress.json"
        self.decisions = json.loads(self.progress_file.read_text(encoding="utf-8")) if self.progress_file.exists() else {}
        self.index = next((ix for ix, it in enumerate(self.entries) if it["file_name"] not in self.decisions), 0)

        self.mixed = False  # 혼합 모드 여부
        self.kind = None  # 지금 그리는 다각형 종류: dot / extra_linear / extra_dot
        self.dots, self.extra, self.current, self.history = [], [], [], []
        self.original = self.photo = self.base = None
        self.offx = self.offy = self.scale = 0
        self.imagew = self.imageh = 0

        root.title("AI Hub 점자블록 라벨링 (+밖 블록 추가): 1 선형, 2 점형, 3 혼합, L/D 추가")
        root.geometry("1250x900")
        root.minsize(850, 650)
        self.status = tk.StringVar(value="")
        tk.Label(root, textvariable=self.status, font=("Malgun Gothic", 11), anchor="w", justify="left").pack(fill="x", padx=10, pady=5)
        tk.Label(
            root,
            text="L=밖 선형 추가  D=밖 점형 추가  (클릭 후 Enter)   →  그다음 1=선형만 2=점형만 3=혼합(점형 클릭, Enter, S)   "
            "Z=다각형 지우기  X=제외  E=내보내기  ←/→ 이동",
            font=("Malgun Gothic", 10),
            anchor="w",
        ).pack(fill="x", padx=10)
        self.canvas = tk.Canvas(root, bg="#222222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Button-3>", lambda e: self.undo_vertex())
        self.canvas.bind("<Button-2>", lambda e: self.undo_vertex())  # macOS 트랙패드 오른쪽 클릭
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        for keys, fn in [
            ("1", lambda: self.quick_save("linear")),
            ("2", lambda: self.quick_save("dot")),
            ("3", self.mixed_mode),
            ("lL", lambda: self.start_draw("extra_linear")),
            ("dD", lambda: self.start_draw("extra_dot")),
            ("sS", self.save_mixed),
            ("xX", lambda: self.quick_save("skip")),
            ("zZ", self.undo_closed),
            ("eE", self.export),
            ("qQ", root.destroy),
        ]:
            for k in keys:
                root.bind(k, lambda e, f=fn: f())
        root.bind("<Return>", lambda e: self.close_polygon())
        root.bind("<BackSpace>", lambda e: self.undo_vertex())
        root.bind("<Right>", lambda e: self.advance(1))
        root.bind("<Left>", lambda e: self.advance(-1))
        root.protocol("WM_DELETE_WINDOW", root.destroy)
        self.load()

    # ---------- 상태 ----------
    def persist(self):
        tmp = self.progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.decisions, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.progress_file)

    def refocus(self):
        self.root.focus_force()
        self.canvas.focus_set()

    def load(self):
        meta = self.entries[self.index]
        with Image.open(self.source / "images_for_cvat" / meta["file_name"]) as im:
            self.original = im.convert("RGB")
        self.imagew, self.imageh = self.original.size
        self.base = union_coco_polygons(self.byid[meta["id"]])
        dec = self.decisions.get(meta["file_name"], {})
        self.extra = [dict(e) for e in dec.get("extra", [])]
        self.dots = [list(map(list, p)) for p in dec.get("dots", [])] if dec.get("mode") == "mixed" else []
        self.mixed, self.kind, self.current, self.history = False, None, [], []
        self.update_status()
        self.redraw()

    def update_status(self, message=""):
        entry = self.entries[self.index]
        tally = Counter(x.get("mode") for x in self.decisions.values())
        n_extra = sum(1 for x in self.decisions.values() if x.get("extra"))
        prior = self.decisions.get(entry["file_name"], {}).get("mode", "미분류")
        drawing = f"그리는 중: {KIND_NAME[self.kind]}" if self.kind else "L/D로 밖 블록 추가, 또는 1/2/3 선택"
        self.status.set(
            f"[{self.index + 1}/{len(self.entries)}] {entry['file_name']}    저장 {sum(tally.values())}/{len(self.entries)} "
            f"(선형 {tally['linear']}, 점형 {tally['dot']}, 혼합 {tally['mixed']}, 제외 {tally['skip']}, 밖 블록 추가 {n_extra})\n"
            f"기존 상태: {prior}  |  추가 블록 {len(self.extra)}개, 혼합 점형 {len(self.dots)}개  |  {drawing}    {message}"
        )

    # ---------- 그리기 ----------
    def to_screen(self, x, y):
        return self.offx + x * self.scale, self.offy + y * self.scale

    def _poly(self, pts, kind):
        flat = [c for x, y in pts for c in self.to_screen(x * self.imagew, y * self.imageh)]
        fill, outline = COLORS[kind]
        self.canvas.create_polygon(flat, fill=fill, outline=outline, stipple="gray50", width=2)

    def redraw(self):
        if self.original is None:
            return
        self.canvas.delete("all")
        cw, ch = self.canvas.winfo_width() or 1000, self.canvas.winfo_height() or 700
        self.scale = min((cw - 30) / self.imagew, (ch - 30) / self.imageh)
        sw, sh = max(1, round(self.imagew * self.scale)), max(1, round(self.imageh * self.scale))
        self.offx, self.offy = (cw - sw) // 2, (ch - sh) // 2
        self.photo = ImageTk.PhotoImage(self.original.resize((sw, sh), Image.Resampling.LANCZOS))
        self.canvas.create_image(self.offx, self.offy, image=self.photo, anchor="nw")
        for p in polygons_of(self.base):
            pts = [c for x, y in list(p.exterior.coords) for c in self.to_screen(x, y)]
            self.canvas.create_polygon(pts, fill="", outline="#FFB020", width=2)
        for e in self.extra:
            self._poly(e["pts"], "extra_" + e["cls"])
        for d in self.dots:
            self._poly(d, "dot")
        if self.current and self.kind:
            color = COLORS[self.kind][1]
            flat = [c for x, y in self.current for c in self.to_screen(x * self.imagew, y * self.imageh)]
            if len(flat) >= 4:
                self.canvas.create_line(flat, fill=color, width=3)
            for x, y in self.current:
                u, v = self.to_screen(x * self.imagew, y * self.imageh)
                self.canvas.create_oval(u - 4, v - 4, u + 4, v + 4, fill=color, outline="white")
        dec = self.decisions.get(self.entries[self.index]["file_name"])
        if dec and dec.get("mode") in ("linear", "dot", "mixed"):
            ex = Counter(e["cls"] for e in dec.get("extra", []))
            text = f"저장됨\n주황 영역: {dec['mode'].upper()}"
            if ex:
                text += f"\n+ 밖 선형 {ex['linear']}개\n+ 밖 점형 {ex['dot']}개"
            self.canvas.create_text(20, 20, text=text, font=("Arial", 17, "bold"), fill="#00DD99", anchor="nw")

    def on_click(self, event):
        if not self.kind:
            return
        x, y = (event.x - self.offx) / self.scale, (event.y - self.offy) / self.scale
        if 0 <= x <= self.imagew and 0 <= y <= self.imageh:
            self.current.append([min(1, max(0, x / self.imagew)), min(1, max(0, y / self.imageh))])
            self.redraw()

    def start_draw(self, kind):
        if self.current:
            self.update_status("그리던 다각형을 Enter로 완성하거나 지운 뒤 바꾸세요")
            return
        self.kind = kind
        self.update_status()
        self.redraw()

    def mixed_mode(self):
        if self.current and self.kind != "dot":
            self.update_status("그리던 다각형을 Enter로 완성하거나 지운 뒤 바꾸세요")
            return
        self.mixed, self.kind = True, "dot"
        self.update_status()
        self.redraw()

    def close_polygon(self):
        if not self.kind:
            return
        if len(self.current) < 3:
            self.update_status("다각형은 최소 3점이 필요해요")
            return
        if self.kind == "dot":
            self.dots.append(self.current)
        else:
            self.extra.append({"cls": self.kind.split("_")[1], "pts": self.current})
        self.history.append(self.kind)
        self.current = []
        self.update_status("완성. 더 그리거나 다음 단계로")
        self.redraw()

    def undo_vertex(self):
        if self.current:
            self.current.pop()
            self.redraw()

    def undo_closed(self):
        if self.current:
            self.current = []
        elif self.history:
            k = self.history.pop()
            (self.dots if k == "dot" else self.extra).pop()
        elif self.extra:
            self.extra.pop()
        elif self.dots:
            self.dots.pop()
        self.update_status()
        self.redraw()

    def advance(self, step):
        new = self.index + step
        if 0 <= new < len(self.entries):
            if self.current or self.history:
                if not messagebox.askyesno("확인", "저장하지 않은 그림이 있습니다. 이동할까요?", parent=self.root):
                    self.refocus()
                    return
            self.index = new
            self.load()

    # ---------- 저장 ----------
    def _commit(self, dec):
        try:
            labels = build_classes(self.base, dec, self.imagew, self.imageh)
            if dec["mode"] != "skip" and not labels_for(labels, self.imagew, self.imageh):
                raise ValueError("유효한 다각형이 없음")
        except Exception as exc:
            messagebox.showerror("확인 필요", str(exc), parent=self.root)
            self.refocus()
            return
        self.decisions[self.entries[self.index]["file_name"]] = dec
        self.persist()
        if self.index < len(self.entries) - 1:
            self.index += 1
        self.load()

    def quick_save(self, mode):
        if self.current:
            self.update_status("그리던 다각형을 Enter로 완성하거나 Z로 지우세요")
            return
        if mode != "skip" and self.base.is_empty and not self.extra:
            messagebox.showerror("라벨 없음", "주황 영역도 추가 블록도 없어요", parent=self.root)
            self.refocus()
            return
        dec = {"mode": mode}
        if mode != "skip" and self.extra:
            dec["extra"] = self.extra
        self._commit(dec)

    def save_mixed(self):
        if self.current:
            messagebox.showwarning("미완성 다각형", "Enter로 현재 다각형을 완성해 주세요.", parent=self.root)
            self.refocus()
            return
        if not self.mixed or not self.dots:
            messagebox.showinfo(
                "S는 혼합 저장용",
                "주황 영역 안에 점형을 그리지 않았어요.\n\n"
                "주황 영역이 전부 선형이면 1, 전부 점형이면 2를 누르세요.\n"
                "(L/D로 추가한 밖 블록은 1/2를 눌러도 같이 저장돼요)\n\n"
                "주황 영역이 섞여 있으면 3 → 점형 클릭 → Enter → S",
                parent=self.root,
            )
            self.refocus()
            return
        dec = {"mode": "mixed", "dots": self.dots}
        if self.extra:
            dec["extra"] = self.extra
        self._commit(dec)

    def export(self):
        imageout, labelout = self.dest / "images" / "all", self.dest / "labels" / "all"
        imageout.mkdir(parents=True, exist_ok=True)
        labelout.mkdir(parents=True, exist_ok=True)
        good = error = 0
        rows = []
        for entry in self.entries:
            fname = entry["file_name"]
            dec = self.decisions.get(fname)
            if dec is None or dec.get("mode") == "skip":
                continue
            try:
                base = union_coco_polygons(self.byid[entry["id"]])
                w, h = entry["width"], entry["height"]
                lines = labels_for(build_classes(base, dec, w, h), w, h)
                if not lines:
                    raise ValueError("Empty label file")
                (labelout / (Path(fname).stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
                shutil.copy2(self.source / "images_for_cvat" / fname, imageout / fname)
                rows.append([fname, dec["mode"], len(dec.get("extra", [])), "OK", len(lines)])
                good += 1
            except Exception as exc:
                rows.append([fname, dec["mode"], len(dec.get("extra", [])), "ERROR: " + str(exc), 0])
                error += 1
        with (self.dest / "export_report.csv").open("w", encoding="utf-8-sig", newline="") as fh:
            cw = csv.writer(fh)
            cw.writerow(["filename", "mode", "extra_polygons", "status", "polygons"])
            cw.writerows(rows)
        (self.dest / "classes.txt").write_text("linear\ndot\n", encoding="utf-8")
        messagebox.showinfo("YOLO 라벨 생성", f"내보내기 완료! 정상 {good}장, 오류 {error}장\n저장: {self.dest.resolve()}", parent=self.root)
        self.refocus()
        self.update_status(f"내보내기 {good}장, 오류 {error}장")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="instances_cvat.json + images_for_cvat/ 가 있는 폴더")
    parser.add_argument("--out", type=Path, default=Path("datasets/aihub_labeled"))
    args = parser.parse_args()
    if tk is None:
        parser.error("tkinter unavailable")
    root = tk.Tk()
    try:
        LabelApp(root, args.input, args.out)
    except Exception as exc:
        root.destroy()
        parser.error(str(exc))
    root.mainloop()


if __name__ == "__main__":
    main()
