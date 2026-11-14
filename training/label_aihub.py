#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AI Hub tactile quick labeling (single class COCO -> linear/dot YOLO polygons).

Designed for EXIF-normalized exports made by prepare_aihub_pilot_fixed.py.
Keys: 1=all linear, 2=all dot, 3=mixed (draw DOT polygons), Enter=close
current DOT polygon, S=save mixed, X=skip, Left/Right=move, Backspace=undo
current vertex, Z=remove last closed DOT polygon, E=export, Q=quit.

Mixed mode: The source AI Hub tactile polygons are the ground-truth region.
Internal holes are cut into multiple hole-free same-class YOLO polygons.
You draw ONLY dot regions. Linear is computed as source minus dot via Shapely.
Requires: pip install pillow shapely
"""
from __future__ import annotations
import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageTk
from shapely.geometry import Polygon, GeometryCollection, LineString
from shapely.ops import unary_union, split

try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    tk = None


def polygons_of(geom):
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == 'Polygon':
        return [geom]
    if hasattr(geom, 'geoms'):
        return [p for child in geom.geoms for p in polygons_of(child)]
    return []


def safe_polygon(pts):
    if len(pts) < 3:
        return None
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if not poly.is_empty else None


def union_coco_polygons(annotations):
    polys = []
    for ann in annotations:
        seg = ann.get('segmentation', [])
        if not isinstance(seg, list):
            continue
        for item in seg:
            if not isinstance(item, list) or len(item) < 6 or len(item) % 2:
                continue
            p = safe_polygon([(item[i], item[i + 1]) for i in range(0, len(item), 2)])
            if p is not None:
                polys.extend(polygons_of(p))
    return unary_union(polys) if polys else GeometryCollection()


def build_two_class(base, decision, width, height):
    mode = decision['mode']
    if mode == 'linear':
        return {'linear': base, 'dot': GeometryCollection()}
    if mode == 'dot':
        return {'linear': GeometryCollection(), 'dot': base}
    if mode == 'skip':
        return {'linear': GeometryCollection(), 'dot': GeometryCollection()}
    if mode != 'mixed':
        raise ValueError('Unknown mode: ' + str(mode))
    marked = []
    for normalized_points in decision.get('dots', []):
        poly = safe_polygon([(x * width, y * height) for x, y in normalized_points])
        if poly is not None:
            marked.extend(polygons_of(poly))
    if not marked:
        raise ValueError('Mixed mode needs at least one completed dot polygon')
    marked_union = unary_union(marked)
    dot = base.intersection(marked_union)
    linear = base.difference(dot)
    if dot.area < 100 or linear.area < 100:
        raise ValueError('One class has too small an area: check dot outline')
    if marked_union.area and dot.area / marked_union.area < .20:
        raise ValueError('Dot polygon falls mostly outside original AI Hub mask; fix image or use CVAT')
    return {'linear': linear, 'dot': dot}



def split_polygon_holes(geom):
    """Convert polygons with holes into multiple hole-free polygons for YOLO-seg.

    A full-width slicing line through each enclosed hole divides a polygon
    without inventing positive label pixels. Resulting parts are emitted as
    separate same-class polygons (semantic union preserves the region).
    """
    stack = list(polygons_of(geom))
    finished = []
    iterations = 0
    while stack:
        p = stack.pop()
        if not p.interiors:
            finished.append(p)
            continue
        iterations += 1
        if iterations > 128:
            raise ValueError('Too many internal holes; please inspect this image in CVAT')
        hole = Polygon(p.interiors[0])
        center = hole.representative_point()
        minx, miny, maxx, maxy = p.bounds
        margin = max(maxx-minx, maxy-miny, 1.0) * 0.05 + 1.0
        probes = [
            LineString([(minx-margin, center.y), (maxx+margin, center.y)]),
            LineString([(center.x, miny-margin), (center.x, maxy+margin)])
        ]
        found = False
        for cutter in probes:
            try:
                parts = [q for q in polygons_of(split(p, cutter)) if q.area > 0.001]
            except Exception:
                continue
            if len(parts) > 1 and sum(len(q.interiors) for q in parts) < len(p.interiors):
                stack.extend(parts)
                found = True
                break
        if not found:
            raise ValueError('Could not split an internal hole automatically; edit this photo in CVAT')
    return finished


def labels_for(class_geoms, w, h):
    lines = []
    for label, cls in [('linear', 0), ('dot', 1)]:
        for p in split_polygon_holes(class_geoms[label]):
            if p.area < 25:
                continue
            vertices = list(p.exterior.coords)[:-1]
            if len(vertices) < 3:
                continue
            coord = []
            for x, y in vertices:
                coord.extend([min(1, max(0, x / w)), min(1, max(0, y / h))])
            lines.append(f'{cls} ' + ' '.join(f'{n:.6f}' for n in coord))
    return lines


class LabelApp:
    def __init__(self, root, source: Path, dest: Path):
        self.root = root
        self.source = source
        self.dest = dest
        self.data = json.loads((source / 'instances_cvat.json').read_text(encoding='utf-8-sig'))
        self.byid = defaultdict(list)
        for a in self.data.get('annotations', []):
            self.byid[a['image_id']].append(a)
        self.entries = [i for i in self.data.get('images', [])
                        if (source / 'images_for_cvat' / i['file_name']).is_file()]
        if not self.entries:
            raise RuntimeError('No images found; check input folder and images_for_cvat')
        self.progress_file = dest / 'quick_label_progress.json'
        dest.mkdir(parents=True, exist_ok=True)
        if self.progress_file.exists():
            self.decisions = json.loads(self.progress_file.read_text(encoding='utf-8'))
        else:
            self.decisions = {}
        self.index = next((ix for ix, item in enumerate(self.entries)
                           if item['file_name'] not in self.decisions), 0)
        self.mode = None
        self.dots = []
        self.current = []
        self.original = None
        self.photo = None
        self.offx = self.offy = self.scale = 0
        self.imagew = self.imageh = 0
        self.base = None

        root.title('AI Hub 점자블록 빠른 라벨링: 1 선형, 2 점형, 3 혼합')
        root.geometry('1250x900')
        root.minsize(850, 650)
        self.status = tk.StringVar(value='')
        tk.Label(root, textvariable=self.status, font=('Malgun Gothic', 11), anchor='w', justify='left').pack(fill='x', padx=10, pady=5)
        tk.Label(root, text='1=선형만   2=점형만   3=혼합(점형만 클릭)   Enter=점형 완성   S=혼합 저장   X=제외   E=전체 내보내기   ←/→ 이동   Z=점형 되돌리기',
                 font=('Malgun Gothic', 10), anchor='w').pack(fill='x', padx=10)
        self.canvas = tk.Canvas(root, bg='#222222', highlightthickness=0)
        self.canvas.pack(fill='both', expand=True, padx=8, pady=8)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Button-3>', lambda e: self.undo_vertex())
        self.canvas.bind('<Configure>', lambda e: self.redraw())
        root.bind('1', lambda e: self.quick_save('linear'))
        root.bind('2', lambda e: self.quick_save('dot'))
        root.bind('3', lambda e: self.mixed_mode())
        root.bind('<Return>', lambda e: self.close_dot())
        root.bind('s', lambda e: self.save_mixed())
        root.bind('S', lambda e: self.save_mixed())
        root.bind('x', lambda e: self.quick_save('skip'))
        root.bind('X', lambda e: self.quick_save('skip'))
        root.bind('<Right>', lambda e: self.advance(1))
        root.bind('<Left>', lambda e: self.advance(-1))
        root.bind('<BackSpace>', lambda e: self.undo_vertex())
        root.bind('z', lambda e: self.undo_closed_dot())
        root.bind('e', lambda e: self.export())
        root.bind('q', lambda e: root.destroy())
        root.protocol('WM_DELETE_WINDOW', root.destroy)
        self.load()

    def persist(self):
        temp = self.progress_file.with_suffix('.tmp')
        temp.write_text(json.dumps(self.decisions, ensure_ascii=False, indent=2), encoding='utf-8')
        temp.replace(self.progress_file)

    def load(self):
        meta = self.entries[self.index]
        imgfile = self.source / 'images_for_cvat' / meta['file_name']
        with Image.open(imgfile) as im:
            self.original = im.convert('RGB')
        self.imagew, self.imageh = self.original.size
        self.base = union_coco_polygons(self.byid[meta['id']])
        self.mode = None
        self.dots = []
        self.current = []
        self.update_status()
        self.redraw()

    def update_status(self, message=''):
        entry = self.entries[self.index]
        tally = Counter(x.get('mode') for x in self.decisions.values())
        saved = sum(tally.values())
        prior = self.decisions.get(entry['file_name'], {}).get('mode', '미분류')
        mode = '혼합: 점형 외곽을 클릭해 Enter, 여러 개면 반복, 마지막 S 저장' if self.mode == 'mixed' else '1/2/3 선택'
        self.status.set(f'[{self.index + 1}/{len(self.entries)}] {entry["file_name"]}    저장 {saved}/{len(self.entries)} '
                        f' (선형 {tally["linear"]}, 점형 {tally["dot"]}, 혼합 {tally["mixed"]}, 제외 {tally["skip"]})\n'
                        f'기존 상태: {prior}    |    {mode}    {message}')

    def redraw(self):
        if self.original is None:
            return
        self.canvas.delete('all')
        cw = self.canvas.winfo_width() or 1000
        ch = self.canvas.winfo_height() or 700
        self.scale = min((cw - 30) / self.imagew, (ch - 30) / self.imageh)
        sw, sh = max(1, round(self.imagew * self.scale)), max(1, round(self.imageh * self.scale))
        self.offx, self.offy = (cw - sw) // 2, (ch - sh) // 2
        thumb = self.original.resize((sw, sh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(thumb)
        self.canvas.create_image(self.offx, self.offy, image=self.photo, anchor='nw')
        # Original tactile polygons in orange, always visible.
        for p in polygons_of(self.base):
            pts = [coord for x,y in list(p.exterior.coords) for coord in self.to_screen(x, y)]
            self.canvas.create_polygon(pts, fill='', outline='#FFB020', width=2)
        dec = self.decisions.get(self.entries[self.index]['file_name'])
        showdots = self.dots if self.mode == 'mixed' else (dec.get('dots', []) if dec and dec.get('mode') == 'mixed' else [])
        for outline in showdots:
            flat = [coord for x,y in outline for coord in self.to_screen(x*self.imagew, y*self.imageh)]
            self.canvas.create_polygon(flat, fill='#7449C8', outline='#BCA8FF', stipple='gray50', width=2)
        if self.current:
            flat = [coord for x,y in self.current for coord in self.to_screen(x*self.imagew, y*self.imageh)]
            if len(flat) >= 4:
                self.canvas.create_line(flat, fill='#BCA8FF', width=3)
            for x,y in self.current:
                u,v = self.to_screen(x*self.imagew, y*self.imageh)
                self.canvas.create_oval(u-4,v-4,u+4,v+4, fill='#BCA8FF', outline='white')
        if dec and dec.get('mode') in ('linear', 'dot'):
            label = 'LINEAR' if dec['mode']=='linear' else 'DOT'
            self.canvas.create_text(20, 20, text=f'저장된 라벨: {label}', font=('Arial', 17, 'bold'), fill='#00DD99', anchor='nw')

    def to_screen(self, x, y):
        return self.offx + x*self.scale, self.offy + y*self.scale

    def on_click(self, event):
        if self.mode != 'mixed':
            return
        x, y = (event.x - self.offx)/self.scale, (event.y-self.offy)/self.scale
        if 0 <= x <= self.imagew and 0 <= y <= self.imageh:
            self.current.append([min(1,max(0,x/self.imagew)), min(1,max(0,y/self.imageh))])
            self.redraw()

    def mixed_mode(self):
        self.mode = 'mixed'
        dec = self.decisions.get(self.entries[self.index]['file_name'], {})
        self.dots = [list(map(list, points)) for points in dec.get('dots', [])] if dec.get('mode') == 'mixed' else []
        self.current = []
        self.update_status()
        self.redraw()

    def undo_vertex(self):
        if self.mode == 'mixed' and self.current:
            self.current.pop()
            self.redraw()

    def undo_closed_dot(self):
        if self.mode == 'mixed' and self.dots:
            self.dots.pop()
            self.redraw()

    def close_dot(self):
        if self.mode != 'mixed':
            return
        if len(self.current) < 3:
            self.update_status('점형 다각형은 최소 3점을 찍어야 함')
            return
        self.dots.append(self.current)
        self.current = []
        self.update_status(f'점형 영역 {len(self.dots)}개. 더 그리거나 S로 저장')
        self.redraw()

    def advance(self, step):
        new = self.index + step
        if 0 <= new < len(self.entries):
            if self.mode == 'mixed' and (self.current or self.dots):
                if not messagebox.askyesno('확인', '저장하지 않은 점형 그림이 있습니다. 이동할까요?'):
                    return
            self.index = new
            self.load()

    def quick_save(self, mode):
        if mode != 'skip' and self.base.is_empty:
            messagebox.showerror('원본 라벨 없음', 'AI Hub tactile 폴리곤이 없습니다. 제외(X)하거나 원본 JSON을 확인하세요.')
            return
        key = self.entries[self.index]['file_name']
        self.decisions[key] = {'mode': mode}
        self.persist()
        self.mode = None
        if self.index < len(self.entries)-1:
            self.index += 1
        self.load()

    def save_mixed(self):
        if self.mode != 'mixed':
            self.update_status('먼저 3을 눌러 혼합 모드로 들어가세요')
            return
        if self.current:
            messagebox.showwarning('완료되지 않은 다각형', 'Enter로 현재 점형 폴리곤을 완료해 주세요.')
            return
        dec = {'mode':'mixed', 'dots': self.dots}
        try:
            labels = build_two_class(self.base, dec, self.imagew, self.imageh)
            if not labels_for(labels, self.imagew, self.imageh):
                raise ValueError('No valid polygon')
        except Exception as exc:
            messagebox.showerror('분할 확인 필요', str(exc))
            return
        self.decisions[self.entries[self.index]['file_name']] = dec
        self.persist()
        self.mode = None
        if self.index < len(self.entries)-1:
            self.index += 1
        self.load()

    def export(self):
        out = self.dest
        imageout = out/'images'/'all'
        labelout = out/'labels'/'all'
        imageout.mkdir(parents=True,exist_ok=True)
        labelout.mkdir(parents=True,exist_ok=True)
        good = error = 0
        rows = []
        for entry in self.entries:
            fname = entry['file_name']
            dec = self.decisions.get(fname)
            if dec is None or dec.get('mode') == 'skip':
                continue
            try:
                base = union_coco_polygons(self.byid[entry['id']])
                if base.is_empty:
                    raise ValueError('Missing tactile polygon')
                w,h = entry['width'], entry['height']
                result = build_two_class(base, dec, w, h)
                lines = labels_for(result, w, h)
                if not lines:
                    raise ValueError('Empty label file')
                (labelout/(Path(fname).stem+'.txt')).write_text('\n'.join(lines)+'\n',encoding='utf-8')
                shutil.copy2(self.source/'images_for_cvat'/fname,imageout/fname)
                rows.append([fname,dec['mode'],'OK',len(lines)])
                good += 1
            except Exception as exc:
                rows.append([fname,dec['mode'],'ERROR: '+str(exc),0])
                error += 1
        with (out/'export_report.csv').open('w',encoding='utf-8-sig',newline='') as fh:
            cw=csv.writer(fh);cw.writerow(['filename','mode','status','polygons']);cw.writerows(rows)
        (out/'classes.txt').write_text('linear\ndot\n',encoding='utf-8')
        message = f'내보내기 완료! 정상 {good}장, 오류 {error}장\n저장: {out.resolve()}\n' + ('export_report.csv의 ERROR를 확인하세요.' if error else '이미지/라벨은 images/all과 labels/all에 있습니다.')
        messagebox.showinfo('YOLO 라벨 생성', message)
        self.update_status(f'내보내기 {good}장, 오류 {error}장')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True, help='Folder containing instances_cvat.json and images_for_cvat/')
    parser.add_argument('--out', type=Path, default=Path('datasets/aihub_labeled'))
    args = parser.parse_args()
    if tk is None:
        parser.error('tkinter unavailable: install a Python distribution with tkinter support')
    root = tk.Tk()
    try:
        LabelApp(root,args.input,args.out)
    except Exception as exc:
        root.destroy()
        parser.error(str(exc))
    root.mainloop()


if __name__ == '__main__':
    main()
