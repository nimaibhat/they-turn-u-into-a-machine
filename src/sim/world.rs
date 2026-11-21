use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrismKind {
    Building,
    Stadium,
    Bridge,
    Other,
}

#[derive(Clone, Debug)]
pub struct Prism {
    pub kind: PrismKind,
    pub base: f64,
    pub top: f64,
    pub ring: Vec<[f64; 2]>,
    pub holes: Vec<Vec<[f64; 2]>>,
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

#[inline]
fn point_in_polygon(poly: &[[f64; 2]], x: f64, y: f64) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = (poly[i][0], poly[i][1]);
        let (xj, yj) = (poly[j][0], poly[j][1]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

impl Prism {
    /// Inside the ring and outside every hole.
    pub fn contains_xy(&self, x: f64, y: f64) -> bool {
        if x < self.min_x || x > self.max_x || y < self.min_y || y > self.max_y {
            return false;
        }
        if !point_in_polygon(&self.ring, x, y) {
            return false;
        }
        !self.holes.iter().any(|h| point_in_polygon(h, x, y))
    }

    /// (signed distance to the footprint boundary, negative inside; closest boundary point;
    /// outward unit normal pointing from solid to free space).
    pub fn dist_xy(&self, x: f64, y: f64) -> (f64, [f64; 2], [f64; 2]) {
        let mut best = f64::INFINITY;
        let mut bp = [x, y];
        let mut be = [1.0, 0.0];
        for poly in std::iter::once(&self.ring).chain(self.holes.iter()) {
            let n = poly.len();
            if n < 2 {
                continue;
            }
            let mut j = n - 1;
            for i in 0..n {
                let (u, v) = (poly[j], poly[i]);
                let ex = v[0] - u[0];
                let yaw_err = v[1] - u[1];
                let l2 = ex * ex + yaw_err * yaw_err;
                let s = if l2 > 0.0 { (((x - u[0]) * ex + (y - u[1]) * yaw_err) / l2).clamp(0.0, 1.0) } else { 0.0 };
                let cx = u[0] + s * ex;
                let cy = u[1] + s * yaw_err;
                let d2 = (x - cx) * (x - cx) + (y - cy) * (y - cy);
                if d2 < best {
                    best = d2;
                    bp = [cx, cy];
                    be = [ex, yaw_err];
                }
                j = i;
            }
        }
        if !best.is_finite() {
            return (f64::INFINITY, [x, y], [1.0, 0.0]);
        }
        let d = best.sqrt();
        let inside = self.contains_xy(x, y);
        let n = if d > 1e-9 {
            let s = if inside { -1.0 } else { 1.0 };
            [s * (x - bp[0]) / d, s * (y - bp[1]) / d]
        } else {
            let l = (be[0] * be[0] + be[1] * be[1]).sqrt().max(1e-12);
            let mut n = [be[1] / l, -be[0] / l];
            if self.contains_xy(bp[0] + n[0] * 1e-4, bp[1] + n[1] * 1e-4) {
                n = [-n[0], -n[1]];
            }
            n
        };
        (if inside { -d } else { d }, bp, n)
    }
}

#[derive(Clone, Debug, Default)]
pub struct Tree {
    pub x: f64,
    pub y: f64,
    /// ground height at the trunk
    pub z: f64,
    /// height above z
    pub height: f64,
    pub crown_r: f64,
    /// crown base above z
    pub crown_base: f64,
    pub trunk_r: f64,
    pub crown_x: f64,
    pub crown_y: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct Bounds {
    pub min_x: f64,
    pub max_x: f64,
    pub min_y: f64,
    pub max_y: f64,
}

#[derive(Clone, Copy, Debug)]
struct CellEntry {
    base: f64,
    top: f64,
    edge_start: u32,
    edge_count: u16,
    prism: u16,
    ref_inside: bool,
    full: bool,
}

const REF_DX: f64 = 0.512_345_67;
const REF_DY: f64 = 0.497_654_32;
const TREE_CELL: f64 = 8.0;
const PRISM_CELL: f64 = 16.0;

pub struct World {
    pub bounds: Bounds,
    pub prisms: Vec<Prism>,
    pub trees: Vec<Tree>,
    pub canopy_density: f64,
    pub canopy_block_t: f64,
    pub max_step: f64,
    pub max_slope_deg: f64,
    // terrain vertex grid, step 1 m; cell (i, j) spans [gx0+i, gx0+i+1] x [gy0+j, gy0+j+1]
    orig_grid_x0: f64,
    orig_grid_y0: f64,
    orig_nx: usize,
    orig_ny: usize,
    pad_left: usize,
    pad_bottom: usize,
    grid_x0: f64,
    grid_y0: f64,
    nx: usize,
    ny: usize,
    heights: Vec<f32>,
    // per cell (cnx = nx-1, cny = ny-1)
    cells_x: usize,
    cells_y: usize,
    cell_hmax: Vec<f32>,
    cell_ztop: Vec<f32>,
    cell_start: Vec<u32>,
    entries: Vec<CellEntry>,
    edges: Vec<[f64; 4]>,
    // trees
    tree_cells_x: usize,
    tree_cells_y: usize,
    tree_start: Vec<u32>,
    tree_idx: Vec<u16>,
    // prisms by bbox
    prism_cells_x: usize,
    prism_cells_y: usize,
    prism_start: Vec<u32>,
    prism_idx: Vec<u16>,
}

fn flat_pairs(v: &serde_json::Value) -> Vec<[f64; 2]> {
    let a = v.as_array().map(|a| a.as_slice()).unwrap_or(&[]);
    let mut out: Vec<[f64; 2]> = a.chunks(2).filter(|c| c.len() == 2).map(|c| [c[0].as_f64().unwrap_or(0.0), c[1].as_f64().unwrap_or(0.0)]).collect();
    if out.len() > 1 && out[0] == out[out.len() - 1] {
        out.pop();
    }
    out
}

#[inline]
fn orientation(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> f64 {
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
}

/// segment c->p crosses edge u->v (half-open convention, orientation 0 counts as positive)
#[inline]
fn segments_cross(c: [f64; 2], p: [f64; 2], u: [f64; 2], v: [f64; 2]) -> bool {
    let o1 = orientation(u, v, c) >= 0.0;
    let o2 = orientation(u, v, p) >= 0.0;
    if o1 == o2 {
        return false;
    }
    let o3 = orientation(c, p, u) >= 0.0;
    let o4 = orientation(c, p, v) >= 0.0;
    o3 != o4
}

/// segment intersects closed axis-aligned box
fn segment_hits_box(ax: f64, ay: f64, bx: f64, by: f64, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
    let mut t0 = 0.0f64;
    let mut t1 = 1.0f64;
    let d = [bx - ax, by - ay];
    let o = [ax, ay];
    let lo = [x0, y0];
    let hi = [x1, y1];
    for k in 0..2 {
        if d[k].abs() < 1e-15 {
            if o[k] < lo[k] || o[k] > hi[k] {
                return false;
            }
        } else {
            let mut ta = (lo[k] - o[k]) / d[k];
            let mut tb = (hi[k] - o[k]) / d[k];
            if ta > tb {
                std::mem::swap(&mut ta, &mut tb);
            }
            t0 = t0.max(ta);
            t1 = t1.min(tb);
            if t0 > t1 {
                return false;
            }
        }
    }
    true
}

impl World {
    pub fn load(dir: &Path) -> Result<World, String> {
        let base = if dir.join("v2").join("world.json").exists() { dir.join("v2") } else { dir.to_path_buf() };
        let text = std::fs::read(base.join("world.json")).map_err(|e| format!("world.json: {e}"))?;
        let j: serde_json::Value = serde_json::from_slice(&text).map_err(|e| format!("world.json parse: {e}"))?;
        let f = |v: &serde_json::Value, k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let b = &j["bounds"];
        let bounds = Bounds { min_x: f(b, "minX", -1e6), max_x: f(b, "maxX", 1e6), min_y: f(b, "minY", -1e6), max_y: f(b, "maxY", 1e6) };
        let r = &j["rules"];
        let track = &j["terrain"];
        let nx = f(track, "nx", 0.0) as usize;
        let ny = f(track, "ny", 0.0) as usize;
        let gx0 = f(track, "minX", 0.0);
        let gy0 = f(track, "minY", 0.0);
        let scale = f(track, "scale", 0.01);
        let step = f(track, "step", 1.0);
        if (step - 1.0).abs() > 1e-9 {
            return Err("terrain step != 1".into());
        }
        if nx < 2 || ny < 2 {
            return Err("terrain too small".into());
        }
        let terrain_file = track.get("file").and_then(|x| x.as_str()).unwrap_or("terrain.i16");
        let raw = std::fs::read(base.join(terrain_file)).map_err(|e| format!("terrain: {e}"))?;
        if raw.len() < nx * ny * 2 {
            return Err("terrain file short".into());
        }
        let h: Vec<f32> = (0..nx * ny).map(|k| (i16::from_le_bytes([raw[2 * k], raw[2 * k + 1]]) as f64 * scale) as f32).collect();

        let mut prisms = Vec::new();
        if let Some(ps) = j["prisms"].as_array() {
            for p in ps {
                let kind = match p.get("kind").and_then(|k| k.as_str()).unwrap_or("") {
                    "building" => PrismKind::Building,
                    "stadium" => PrismKind::Stadium,
                    "bridge" => PrismKind::Bridge,
                    _ => PrismKind::Other,
                };
                let ring = flat_pairs(&p["ring"]);
                if ring.len() < 3 {
                    continue;
                }
                let holes: Vec<Vec<[f64; 2]>> = p["holes"].as_array().map(|hs| hs.iter().map(flat_pairs).filter(|h| h.len() >= 3).collect()).unwrap_or_default();
                let (mut a, mut bx, mut c, mut d) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
                for q in &ring {
                    a = a.min(q[0]);
                    bx = bx.max(q[0]);
                    c = c.min(q[1]);
                    d = d.max(q[1]);
                }
                prisms.push(Prism { kind, base: f(p, "base", 0.0), top: f(p, "top", 0.0), ring, holes, min_x: a, max_x: bx, min_y: c, max_y: d });
            }
        }
        let mut trees = Vec::new();
        let tj = &j["trees"];
        if let (Some(fields), Some(data)) = (tj["fields"].as_array(), tj["data"].as_array()) {
            let names: Vec<&str> = fields.iter().map(|x| x.as_str().unwrap_or("")).collect();
            let nf = names.len();
            let col = |n: &str| names.iter().position(|x| *x == n);
            let (cx, cy, cz, ch, ccr, ccb, ctr, ccx, ccy) = (col("x"), col("y"), col("z"), col("height"), col("crownR"), col("crownBase"), col("trunkR"), col("crownX"), col("crownY"));
            if nf > 0 {
                for row in data.chunks(nf) {
                    if row.len() < nf {
                        break;
                    }
                    let g = |c: Option<usize>, d: f64| c.and_then(|i| row[i].as_f64()).unwrap_or(d);
                    let x = g(cx, 0.0);
                    let y = g(cy, 0.0);
                    trees.push(Tree { x, y, z: g(cz, 0.0), height: g(ch, 0.0), crown_r: g(ccr, 0.0), crown_base: g(ccb, 0.0), trunk_r: g(ctr, 0.0), crown_x: g(ccx, x), crown_y: g(ccy, y) });
                }
            }
        }
        // pad the grid (clamped heights) so every prism and tree footprint lies inside it
        let (mut ex0, mut ex1, mut ey0, mut ey1) = (gx0, gx0 + (nx - 1) as f64, gy0, gy0 + (ny - 1) as f64);
        for p in &prisms {
            ex0 = ex0.min(p.min_x);
            ex1 = ex1.max(p.max_x);
            ey0 = ey0.min(p.min_y);
            ey1 = ey1.max(p.max_y);
        }
        for t in &trees {
            let r = t.crown_r.max(t.trunk_r);
            ex0 = ex0.min(t.x.min(t.crown_x) - r);
            ex1 = ex1.max(t.x.max(t.crown_x) + r);
            ey0 = ey0.min(t.y.min(t.crown_y) - r);
            ey1 = ey1.max(t.y.max(t.crown_y) + r);
        }
        let pl = ((gx0 - ex0).max(0.0).ceil() as usize) + 2;
        let pb = ((gy0 - ey0).max(0.0).ceil() as usize) + 2;
        let pr = ((ex1 - (gx0 + (nx - 1) as f64)).max(0.0).ceil() as usize) + 2;
        let pt = ((ey1 - (gy0 + (ny - 1) as f64)).max(0.0).ceil() as usize) + 2;
        let (onx, ony) = (nx, ny);
        let nx2 = nx + pl + pr;
        let ny2 = ny + pb + pt;
        let mut h2 = vec![0f32; nx2 * ny2];
        for j in 0..ny2 {
            let sj = (j as i64 - pb as i64).clamp(0, ony as i64 - 1) as usize;
            for i in 0..nx2 {
                let si = (i as i64 - pl as i64).clamp(0, onx as i64 - 1) as usize;
                h2[j * nx2 + i] = h[sj * onx + si];
            }
        }
        let mut w = World {
            bounds,
            orig_grid_x0: gx0,
            orig_grid_y0: gy0,
            orig_nx: onx,
            orig_ny: ony,
            pad_left: pl,
            pad_bottom: pb,
            prisms,
            trees,
            canopy_density: f(r, "canopyDensity", 0.35),
            canopy_block_t: f(r, "canopyBlockT", 0.25),
            max_step: f(r, "maxStep", 1.5),
            max_slope_deg: f(r, "maxSlopeDeg", 50.0),
            grid_x0: gx0 - pl as f64,
            grid_y0: gy0 - pb as f64,
            nx: nx2,
            ny: ny2,
            heights: h2,
            cells_x: 0,
            cells_y: 0,
            cell_hmax: Vec::new(),
            cell_ztop: Vec::new(),
            cell_start: Vec::new(),
            entries: Vec::new(),
            edges: Vec::new(),
            tree_cells_x: 0,
            tree_cells_y: 0,
            tree_start: Vec::new(),
            tree_idx: Vec::new(),
            prism_cells_x: 0,
            prism_cells_y: 0,
            prism_start: Vec::new(),
            prism_idx: Vec::new(),
        };
        w.build_index();
        Ok(w)
    }

    pub fn empty() -> World {
        World {
            bounds: Bounds { min_x: -1e6, max_x: 1e6, min_y: -1e6, max_y: 1e6 },
            prisms: Vec::new(),
            trees: Vec::new(),
            canopy_density: 0.35,
            canopy_block_t: 0.25,
            max_step: 1.5,
            max_slope_deg: 50.0,
            orig_grid_x0: 0.0,
            orig_grid_y0: 0.0,
            orig_nx: 0,
            orig_ny: 0,
            pad_left: 0,
            pad_bottom: 0,
            grid_x0: 0.0,
            grid_y0: 0.0,
            nx: 0,
            ny: 0,
            heights: Vec::new(),
            cells_x: 0,
            cells_y: 0,
            cell_hmax: Vec::new(),
            cell_ztop: Vec::new(),
            cell_start: vec![0],
            entries: Vec::new(),
            edges: Vec::new(),
            tree_cells_x: 0,
            tree_cells_y: 0,
            tree_start: vec![0],
            tree_idx: Vec::new(),
            prism_cells_x: 0,
            prism_cells_y: 0,
            prism_start: vec![0],
            prism_idx: Vec::new(),
        }
    }

    fn build_index(&mut self) {
        let cnx = self.nx - 1;
        let cny = self.ny - 1;
        self.cells_x = cnx;
        self.cells_y = cny;
        let nc = cnx * cny;
        let mut height_max = vec![f32::NEG_INFINITY; nc];
        for j in 0..cny {
            for i in 0..cnx {
                let k = j * self.nx + i;
                height_max[j * cnx + i] = self.heights[k].max(self.heights[k + 1]).max(self.heights[k + self.nx]).max(self.heights[k + self.nx + 1]);
            }
        }
        self.cell_hmax = height_max;
        // (cell, prism, edge) triples for boundary cells
        let mut bnd: Vec<(u32, u16, [f64; 4])> = Vec::new();
        let mut full: Vec<(u32, u16)> = Vec::new();
        let (gx0, gy0) = (self.grid_x0, self.grid_y0);
        let ci = |x: f64| ((x - gx0).floor() as i64).clamp(0, cnx as i64 - 1) as usize;
        let cj = |y: f64| ((y - gy0).floor() as i64).clamp(0, cny as i64 - 1) as usize;
        let mut mark = vec![u32::MAX; nc];
        for (pi, p) in self.prisms.iter().enumerate() {
            let pi16 = pi as u16;
            for poly in std::iter::once(&p.ring).chain(p.holes.iter()) {
                let n = poly.len();
                let mut jv = n - 1;
                for iv in 0..n {
                    let (u, v) = (poly[jv], poly[iv]);
                    jv = iv;
                    let pad = 1e-7;
                    let (i0, i1) = (ci(u[0].min(v[0]) - pad), ci(u[0].max(v[0]) + pad));
                    let (j0, j1) = (cj(u[1].min(v[1]) - pad), cj(u[1].max(v[1]) + pad));
                    for jj in j0..=j1 {
                        for ii in i0..=i1 {
                            let x0 = gx0 + ii as f64;
                            let y0 = gy0 + jj as f64;
                            if segment_hits_box(u[0], u[1], v[0], v[1], x0 - pad, y0 - pad, x0 + 1.0 + pad, y0 + 1.0 + pad) {
                                let c = (jj * cnx + ii) as u32;
                                bnd.push((c, pi16, [u[0], u[1], v[0], v[1]]));
                                mark[c as usize] = pi as u32;
                            }
                        }
                    }
                }
            }
            // interior fill by scanline at cell-centre rows: cells whose centre is inside and not boundary
            let (j0, j1) = (cj(p.min_y), cj(p.max_y));
            let mut xs: Vec<f64> = Vec::new();
            for jj in j0..=j1 {
                let yc = gy0 + jj as f64 + 0.5;
                xs.clear();
                for poly in std::iter::once(&p.ring).chain(p.holes.iter()) {
                    let n = poly.len();
                    let mut jv = n - 1;
                    for iv in 0..n {
                        let (a, b) = (poly[iv], poly[jv]);
                        if (a[1] > yc) != (b[1] > yc) {
                            xs.push((b[0] - a[0]) * (yc - a[1]) / (b[1] - a[1]) + a[0]);
                        }
                        jv = iv;
                    }
                }
                xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
                for pair in xs.chunks(2) {
                    if pair.len() < 2 {
                        break;
                    }
                    let (xa, xb) = (pair[0], pair[1]);
                    // cells whose centre lies in [xa, xb]
                    let ia = ((xa - gx0 - 0.5).ceil() as i64).max(0);
                    let ib = ((xb - gx0 - 0.5).floor() as i64).min(cnx as i64 - 1);
                    let mut ii = ia;
                    while ii <= ib {
                        let c = (jj * cnx + ii as usize) as u32;
                        if mark[c as usize] != pi as u32 {
                            full.push((c, pi16));
                        }
                        ii += 1;
                    }
                }
            }
        }
        bnd.sort_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)));
        full.sort();
        full.dedup();
        // merge into per-cell entries
        let mut counts = vec![0u32; nc + 1];
        let mut entries: Vec<(u32, CellEntry)> = Vec::new();
        let mut edges: Vec<[f64; 4]> = Vec::with_capacity(bnd.len());
        let mut k = 0;
        while k < bnd.len() {
            let (c, pi, _) = bnd[k];
            let e0 = edges.len() as u32;
            while k < bnd.len() && bnd[k].0 == c && bnd[k].1 == pi {
                edges.push(bnd[k].2);
                k += 1;
            }
            let p = &self.prisms[pi as usize];
            let ii = c as usize % cnx;
            let jj = c as usize / cnx;
            let rx = gx0 + ii as f64 + REF_DX;
            let ry = gy0 + jj as f64 + REF_DY;
            entries.push((c, CellEntry { base: p.base, top: p.top, edge_start: e0, edge_count: (edges.len() as u32 - e0) as u16, prism: pi, ref_inside: p.contains_xy(rx, ry), full: false }));
        }
        for &(c, pi) in &full {
            let p = &self.prisms[pi as usize];
            entries.push((c, CellEntry { base: p.base, top: p.top, edge_start: 0, edge_count: 0, prism: pi, ref_inside: true, full: true }));
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.prism.cmp(&b.1.prism)));
        for e in &entries {
            counts[e.0 as usize + 1] += 1;
        }
        for c in 0..nc {
            counts[c + 1] += counts[c];
        }
        let mut z_top = vec![f32::NEG_INFINITY; nc];
        for e in &entries {
            let t = e.1.top as f32;
            if t > z_top[e.0 as usize] {
                z_top[e.0 as usize] = t;
            }
        }
        self.cell_ztop = z_top;
        self.cell_start = counts;
        self.entries = entries.into_iter().map(|e| e.1).collect();
        self.edges = edges;

        // trees
        let tnx = ((cnx as f64) / TREE_CELL).ceil() as usize;
        let tny = ((cny as f64) / TREE_CELL).ceil() as usize;
        let mut tl: Vec<(u32, u16)> = Vec::new();
        for (ti, t) in self.trees.iter().enumerate() {
            let x0 = (t.x - t.trunk_r).min(t.crown_x - t.crown_r);
            let x1 = (t.x + t.trunk_r).max(t.crown_x + t.crown_r);
            let y0 = (t.y - t.trunk_r).min(t.crown_y - t.crown_r);
            let y1 = (t.y + t.trunk_r).max(t.crown_y + t.crown_r);
            let tc = |v: f64, o: f64, n: usize| (((v - o) / TREE_CELL).floor() as i64).clamp(0, n as i64 - 1) as usize;
            for jj in tc(y0, gy0, tny)..=tc(y1, gy0, tny) {
                for ii in tc(x0, gx0, tnx)..=tc(x1, gx0, tnx) {
                    tl.push(((jj * tnx + ii) as u32, ti as u16));
                }
            }
        }
        tl.sort();
        let mut ts = vec![0u32; tnx * tny + 1];
        for e in &tl {
            ts[e.0 as usize + 1] += 1;
        }
        for c in 0..tnx * tny {
            ts[c + 1] += ts[c];
        }
        self.tree_cells_x = tnx;
        self.tree_cells_y = tny;
        self.tree_start = ts;
        self.tree_idx = tl.into_iter().map(|e| e.1).collect();

        // prisms by bbox
        let pnx = ((cnx as f64) / PRISM_CELL).ceil() as usize;
        let pny = ((cny as f64) / PRISM_CELL).ceil() as usize;
        let mut pl: Vec<(u32, u16)> = Vec::new();
        for (pi, p) in self.prisms.iter().enumerate() {
            let pc = |v: f64, o: f64, n: usize| (((v - o) / PRISM_CELL).floor() as i64).clamp(0, n as i64 - 1) as usize;
            for jj in pc(p.min_y, gy0, pny)..=pc(p.max_y, gy0, pny) {
                for ii in pc(p.min_x, gx0, pnx)..=pc(p.max_x, gx0, pnx) {
                    pl.push(((jj * pnx + ii) as u32, pi as u16));
                }
            }
        }
        pl.sort();
        let mut ps = vec![0u32; pnx * pny + 1];
        for e in &pl {
            ps[e.0 as usize + 1] += 1;
        }
        for c in 0..pnx * pny {
            ps[c + 1] += ps[c];
        }
        self.prism_cells_x = pnx;
        self.prism_cells_y = pny;
        self.prism_start = ps;
        self.prism_idx = pl.into_iter().map(|e| e.1).collect();
    }

    #[inline]
    fn cell_of(&self, x: f64, y: f64) -> Option<usize> {
        if self.cells_x == 0 {
            return None;
        }
        let fx = x - self.grid_x0;
        let fy = y - self.grid_y0;
        if !(fx >= 0.0 && fy >= 0.0) {
            return None;
        }
        let i = fx as usize;
        let j = fy as usize;
        if i >= self.cells_x || j >= self.cells_y {
            return None;
        }
        Some(j * self.cells_x + i)
    }

    #[inline]
    fn cell_entries(&self, c: usize) -> &[CellEntry] {
        &self.entries[self.cell_start[c] as usize..self.cell_start[c + 1] as usize]
    }

    /// point-in-footprint using only the edges stored for this cell; (x, y) must lie in cell c
    #[inline]
    fn inside_local(&self, c: usize, e: &CellEntry, x: f64, y: f64) -> bool {
        if e.full {
            return true;
        }
        let ii = c % self.cells_x;
        let jj = c / self.cells_x;
        let r = [self.grid_x0 + ii as f64 + REF_DX, self.grid_y0 + jj as f64 + REF_DY];
        let p = [x, y];
        let mut inside = e.ref_inside;
        for ed in &self.edges[e.edge_start as usize..e.edge_start as usize + e.edge_count as usize] {
            if segments_cross(r, p, [ed[0], ed[1]], [ed[2], ed[3]]) {
                inside = !inside;
            }
        }
        inside
    }

    /// Bilinear terrain height (clamped to the grid edge outside the grid).
    #[inline]
    pub fn terrain_z(&self, x: f64, y: f64) -> f64 {
        if self.nx < 2 {
            return 0.0;
        }
        let fx = (x - self.grid_x0).clamp(0.0, (self.nx - 1) as f64);
        let fy = (y - self.grid_y0).clamp(0.0, (self.ny - 1) as f64);
        let i = (fx as usize).min(self.nx - 2);
        let j = (fy as usize).min(self.ny - 2);
        let u = fx - i as f64;
        let v = fy - j as f64;
        let k = j * self.nx + i;
        let h00 = self.heights[k] as f64;
        let h10 = self.heights[k + 1] as f64;
        let h01 = self.heights[k + self.nx] as f64;
        let h11 = self.heights[k + self.nx + 1] as f64;
        h00 + (h10 - h00) * u + (h01 - h00) * v + (h00 - h10 - h01 + h11) * u * v
    }

    /// Terrain height at vertex (i, j) of the file grid.
    pub fn terrain_vertex(&self, i: usize, j: usize) -> f64 {
        self.heights[(j + self.pad_bottom) * self.nx + i + self.pad_left] as f64
    }

    /// (minX, minY, nx, ny) of the terrain file grid (step 1 m).
    pub fn terrain_grid(&self) -> (f64, f64, usize, usize) {
        (self.orig_grid_x0, self.orig_grid_y0, self.orig_nx, self.orig_ny)
    }

    /// Highest walkable surface (terrain or prism top) at or below z (1e-6 tolerance).
    pub fn floor_below(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut best = self.terrain_z(x, y);
        if let Some(c) = self.cell_of(x, y) {
            for e in self.cell_entries(c) {
                if e.top <= z + 1e-6 && e.top > best && self.inside_local(c, e, x, y) {
                    best = e.top;
                }
            }
        }
        best
    }

    /// Lowest prism base strictly above z, or INFINITY.
    pub fn ceiling_above(&self, x: f64, y: f64, z: f64) -> f64 {
        let mut best = f64::INFINITY;
        if let Some(c) = self.cell_of(x, y) {
            for e in self.cell_entries(c) {
                if e.base > z && e.base < best && self.inside_local(c, e, x, y) {
                    best = e.base;
                }
            }
        }
        best
    }

    /// Inside terrain (z < ground) or inside a prism (base <= z < top).
    pub fn solid(&self, x: f64, y: f64, z: f64) -> bool {
        if z < self.terrain_z(x, y) {
            return true;
        }
        if let Some(c) = self.cell_of(x, y) {
            if (z as f32) > self.cell_ztop[c] + 0.01 {
                return false;
            }
            for e in self.cell_entries(c) {
                if z >= e.base && z < e.top && self.inside_local(c, e, x, y) {
                    return true;
                }
            }
        }
        false
    }

    /// (base, top) of every prism whose footprint contains (x, y).
    pub fn columns(&self, x: f64, y: f64, out: &mut Vec<(f64, f64)>) {
        out.clear();
        if let Some(c) = self.cell_of(x, y) {
            for e in self.cell_entries(c) {
                if self.inside_local(c, e, x, y) {
                    out.push((e.base, e.top));
                }
            }
        }
    }

    /// Indices of prisms containing (x, y) via callback.
    pub fn prisms_at(&self, x: f64, y: f64, mut f: impl FnMut(usize, &Prism)) {
        if let Some(c) = self.cell_of(x, y) {
            for e in self.cell_entries(c) {
                if self.inside_local(c, e, x, y) {
                    f(e.prism as usize, &self.prisms[e.prism as usize]);
                }
            }
        }
    }

    /// Visits every prism whose footprint comes within r of (x, y) (inside counts), each once.
    pub fn prisms_near(&self, x: f64, y: f64, r: f64, mut f: impl FnMut(usize, &Prism)) {
        if self.prism_cells_x == 0 {
            return;
        }
        let pc = |v: f64, o: f64, n: usize| (((v - o) / PRISM_CELL).floor() as i64).clamp(0, n as i64 - 1) as usize;
        let (qx0, qx1, qy0, qy1) = (x - r, x + r, y - r, y + r);
        for jj in pc(qy0, self.grid_y0, self.prism_cells_y)..=pc(qy1, self.grid_y0, self.prism_cells_y) {
            for ii in pc(qx0, self.grid_x0, self.prism_cells_x)..=pc(qx1, self.grid_x0, self.prism_cells_x) {
                let c = jj * self.prism_cells_x + ii;
                for &pi in &self.prism_idx[self.prism_start[c] as usize..self.prism_start[c + 1] as usize] {
                    let p = &self.prisms[pi as usize];
                    if p.min_x > qx1 || p.max_x < qx0 || p.min_y > qy1 || p.max_y < qy0 {
                        continue;
                    }
                    // dedupe: handle only in the cell holding the low corner of the overlap
                    if pc(qx0.max(p.min_x), self.grid_x0, self.prism_cells_x) != ii || pc(qy0.max(p.min_y), self.grid_y0, self.prism_cells_y) != jj {
                        continue;
                    }
                    let dx = (p.min_x - x).max(0.0).max(x - p.max_x);
                    let dy = (p.min_y - y).max(0.0).max(y - p.max_y);
                    if dx * dx + dy * dy > r * r {
                        continue;
                    }
                    if p.dist_xy(x, y).0 <= r {
                        f(pi as usize, p);
                    }
                }
            }
        }
    }

    /// t in [0,1] of the first hit of segment a->b with terrain or a prism.
    pub fn ray(&self, a: [f64; 3], b: [f64; 3]) -> Option<f64> {
        if self.solid(a[0], a[1], a[2]) {
            return Some(0.0);
        }
        if self.cells_x == 0 {
            // flat ground at z = 0
            if b[2] < 0.0 && a[2] >= 0.0 {
                return Some((a[2] / (a[2] - b[2])).clamp(0.0, 1.0));
            }
            return None;
        }
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        // clip xy to grid rect
        let (x0, x1) = (self.grid_x0, self.grid_x0 + self.cells_x as f64);
        let (y0, y1) = (self.grid_y0, self.grid_y0 + self.cells_y as f64);
        let mut ts = 0.0f64;
        let mut te = 1.0f64;
        for k in 0..2 {
            let (o, dk, lo, hi) = if k == 0 { (a[0], d[0], x0, x1) } else { (a[1], d[1], y0, y1) };
            if dk.abs() < 1e-12 {
                if o < lo || o >= hi {
                    return None;
                }
            } else {
                let mut t0 = (lo - o) / dk;
                let mut t1 = (hi - o) / dk;
                if t0 > t1 {
                    std::mem::swap(&mut t0, &mut t1);
                }
                ts = ts.max(t0);
                te = te.min(t1);
            }
        }
        if ts > te {
            return None;
        }
        let px = a[0] + d[0] * ts;
        let py = a[1] + d[1] * ts;
        let mut i = ((px - self.grid_x0).floor() as i64).clamp(0, self.cells_x as i64 - 1);
        let mut j = ((py - self.grid_y0).floor() as i64).clamp(0, self.cells_y as i64 - 1);
        let (sx, tdx, mut tmx) = if d[0] > 1e-12 {
            (1i64, 1.0 / d[0], (self.grid_x0 + (i + 1) as f64 - a[0]) / d[0])
        } else if d[0] < -1e-12 {
            (-1i64, -1.0 / d[0], (self.grid_x0 + i as f64 - a[0]) / d[0])
        } else {
            (0, f64::INFINITY, f64::INFINITY)
        };
        let (sy, tdy, mut tmy) = if d[1] > 1e-12 {
            (1i64, 1.0 / d[1], (self.grid_y0 + (j + 1) as f64 - a[1]) / d[1])
        } else if d[1] < -1e-12 {
            (-1i64, -1.0 / d[1], (self.grid_y0 + j as f64 - a[1]) / d[1])
        } else {
            (0, f64::INFINITY, f64::INFINITY)
        };
        let mut t0 = ts;
        loop {
            let t1 = tmx.min(tmy).min(te);
            let c = j as usize * self.cells_x + i as usize;
            let za = a[2] + d[2] * t0;
            let zb = a[2] + d[2] * t1;
            let z_min = za.min(zb);
            if let Some(t) = self.cell_hit(c, i as usize, j as usize, a, d, t0, t1, z_min) {
                return Some(t.clamp(0.0, 1.0));
            }
            if t1 >= te {
                return None;
            }
            if tmx < tmy {
                i += sx;
                tmx += tdx;
                if i < 0 || i >= self.cells_x as i64 {
                    return None;
                }
            } else {
                j += sy;
                tmy += tdy;
                if j < 0 || j >= self.cells_y as i64 {
                    return None;
                }
            }
            t0 = t1;
        }
    }

    #[inline]
    fn cell_hit(&self, c: usize, i: usize, j: usize, a: [f64; 3], d: [f64; 3], t0: f64, t1: f64, z_min: f64) -> Option<f64> {
        let mut best = f64::INFINITY;
        // terrain
        if z_min <= self.cell_hmax[c] as f64 + 1e-6 {
            let k = j * self.nx + i;
            let h00 = self.heights[k] as f64;
            let h10 = self.heights[k + 1] as f64;
            let h01 = self.heights[k + self.nx] as f64;
            let h11 = self.heights[k + self.nx + 1] as f64;
            let ua = a[0] - (self.grid_x0 + i as f64);
            let va = a[1] - (self.grid_y0 + j as f64);
            let (bu, bv, buv) = (h10 - h00, h01 - h00, h00 - h10 - h01 + h11);
            // f(t) = h(u(t), v(t)) - z(t) = A + B t + C t^2
            let ca = h00 + bu * ua + bv * va + buv * ua * va - a[2];
            let cb = bu * d[0] + bv * d[1] + buv * (ua * d[1] + va * d[0]) - d[2];
            let cc = buv * d[0] * d[1];
            let fv = |t: f64| ca + cb * t + cc * t * t;
            if fv(t0) >= 0.0 {
                best = t0;
            } else {
                let mut track = f64::INFINITY;
                if cc.abs() < 1e-12 {
                    if cb.abs() > 1e-15 {
                        let t = -ca / cb;
                        if t >= t0 && t <= t1 {
                            track = t;
                        }
                    }
                } else {
                    let discriminant = cb * cb - 4.0 * cc * ca;
                    if discriminant >= 0.0 {
                        let sqrt_disc = discriminant.sqrt();
                        let q = -0.5 * (cb + cb.signum() * sqrt_disc);
                        let r1 = q / cc;
                        let r2 = if q != 0.0 { ca / q } else { r1 };
                        for t in [r1, r2] {
                            if t >= t0 && t <= t1 && t < track {
                                track = t;
                            }
                        }
                    }
                }
                if track.is_infinite() && fv(t1) >= 0.0 {
                    track = t1;
                }
                best = track;
            }
        }
        // prisms
        if z_min < self.cell_ztop[c] as f64 + 1e-6 {
            for e in self.cell_entries(c) {
                // walls
                for ed in &self.edges[e.edge_start as usize..e.edge_start as usize + e.edge_count as usize] {
                    let ex = ed[2] - ed[0];
                    let yaw_err = ed[3] - ed[1];
                    let denom = d[0] * yaw_err - d[1] * ex;
                    if denom.abs() < 1e-14 {
                        continue;
                    }
                    let wx = ed[0] - a[0];
                    let wy = ed[1] - a[1];
                    let t = (wx * yaw_err - wy * ex) / denom;
                    if t < t0 - 1e-9 || t > t1 + 1e-9 || t >= best {
                        continue;
                    }
                    let s = (wx * d[1] - wy * d[0]) / denom;
                    if !(-1e-9..=1.0 + 1e-9).contains(&s) {
                        continue;
                    }
                    let z = a[2] + d[2] * t;
                    if z >= e.base && z < e.top {
                        best = t;
                    }
                }
                // caps
                if d[2] != 0.0 {
                    let h = if d[2] < 0.0 { e.top } else { e.base };
                    let t = (h - a[2]) / d[2];
                    if t >= t0 - 1e-9 && t <= t1 + 1e-9 && t < best {
                        let x = a[0] + d[0] * t;
                        let y = a[1] + d[1] * t;
                        if self.inside_local(c, e, x, y) {
                            best = t;
                        }
                    }
                }
            }
        }
        if best.is_finite() {
            Some(best)
        } else {
            None
        }
    }

    /// No terrain or prism on the segment.
    pub fn sight_no_trees(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        self.ray(a, b).is_none()
    }

    /// Server-rule guess: buildings, terrain and trunks block; canopy blocks when transmittance < canopyBlockT.
    pub fn sight(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        if self.ray(a, b).is_some() {
            return false;
        }
        let (trunk, len) = self.trees_on(a, b, true, self.canopy_block_len());
        !trunk && !self.canopy_blocks(len)
    }

    /// Like sight() but canopy ignored (vision inside near_range): buildings, terrain and trunks block.
    pub fn sight_near(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        if self.ray(a, b).is_some() {
            return false;
        }
        !self.trees_on(a, b, true, f64::INFINITY).0
    }

    /// Any tree trunk on the segment.
    pub fn trunk_blocks(&self, a: [f64; 3], b: [f64; 3]) -> bool {
        self.trees_on(a, b, true, f64::INFINITY).0
    }

    /// Metres of the segment inside tree crowns (ellipsoids, summed over overlapping crowns).
    pub fn canopy_len(&self, a: [f64; 3], b: [f64; 3]) -> f64 {
        self.trees_on(a, b, false, f64::INFINITY).1
    }

    // ---- canopy rule (guess) ----
    /// Transmittance through len metres of canopy.
    pub fn canopy_transmittance(&self, len: f64) -> f64 {
        (-self.canopy_density * len).exp()
    }
    pub fn canopy_blocks(&self, len: f64) -> bool {
        self.canopy_transmittance(len) < self.canopy_block_t
    }
    /// Canopy length at which sight is lost.
    pub fn canopy_block_len(&self) -> f64 {
        if self.canopy_density <= 0.0 || self.canopy_block_t <= 0.0 {
            f64::INFINITY
        } else {
            -self.canopy_block_t.ln() / self.canopy_density
        }
    }

    /// Crown of a tree: ellipsoid centred at (crown_x, crown_y, z + (crown_base + height)/2),
    /// horizontal radius crown_r, vertical half-axis (height - crown_base)/2.
    /// Trunk: vertical cylinder at (x, y), radius trunk_r, from z to z + crown_base.
    /// Returns (trunk hit, canopy length); stops early once canopy length exceeds stop_len.
    fn trees_on(&self, a: [f64; 3], b: [f64; 3], trunks: bool, stop_len: f64) -> (bool, f64) {
        if self.trees.is_empty() || self.tree_cells_x == 0 {
            return (false, 0.0);
        }
        let d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let segment_len = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let inv = 1.0 / TREE_CELL;
        let fx = (a[0] - self.grid_x0) * inv;
        let fy = (a[1] - self.grid_y0) * inv;
        let dx = d[0] * inv;
        let dy = d[1] * inv;
        let n_x = self.tree_cells_x as i64;
        let n_y = self.tree_cells_y as i64;
        let mut i = fx.floor() as i64;
        let mut j = fy.floor() as i64;
        let (sx, tdx, mut tmx) = if dx > 1e-12 {
            (1i64, 1.0 / dx, ((i + 1) as f64 - fx) / dx)
        } else if dx < -1e-12 {
            (-1i64, -1.0 / dx, (i as f64 - fx) / dx)
        } else {
            (0, f64::INFINITY, f64::INFINITY)
        };
        let (sy, tdy, mut tmy) = if dy > 1e-12 {
            (1i64, 1.0 / dy, ((j + 1) as f64 - fy) / dy)
        } else if dy < -1e-12 {
            (-1i64, -1.0 / dy, (j as f64 - fy) / dy)
        } else {
            (0, f64::INFINITY, f64::INFINITY)
        };
        let mut t0 = 0.0f64;
        let mut len = 0.0f64;
        let mut trunk_hit = false;
        loop {
            let t1 = tmx.min(tmy).min(1.0);
            if i >= 0 && j >= 0 && i < n_x && j < n_y {
                let c = (j * n_x + i) as usize;
                for &ti in &self.tree_idx[self.tree_start[c] as usize..self.tree_start[c + 1] as usize] {
                    let t = &self.trees[ti as usize];
                    // crown, clipped to this cell's t-range so each tree is counted once
                    let rz = 0.5 * (t.height - t.crown_base);
                    if t.crown_r > 0.0 && rz > 0.0 {
                        let cz = t.z + 0.5 * (t.crown_base + t.height);
                        let ox = (a[0] - t.crown_x) / t.crown_r;
                        let oy = (a[1] - t.crown_y) / t.crown_r;
                        let oz = (a[2] - cz) / rz;
                        let ux = d[0] / t.crown_r;
                        let uy = d[1] / t.crown_r;
                        let uz = d[2] / rz;
                        let qa = ux * ux + uy * uy + uz * uz;
                        let qb = 2.0 * (ox * ux + oy * uy + oz * uz);
                        let qc = ox * ox + oy * oy + oz * oz - 1.0;
                        if qa > 0.0 {
                            let discriminant = qb * qb - 4.0 * qa * qc;
                            if discriminant > 0.0 {
                                let sqrt_disc = discriminant.sqrt();
                                let ta = ((-qb - sqrt_disc) / (2.0 * qa)).max(t0);
                                let tb = ((-qb + sqrt_disc) / (2.0 * qa)).min(t1);
                                if tb > ta {
                                    len += (tb - ta) * segment_len;
                                }
                            }
                        }
                    }
                    if trunks && !trunk_hit && t.trunk_r > 0.0 && t.crown_base > 0.0 {
                        let ox = a[0] - t.x;
                        let oy = a[1] - t.y;
                        let qa = d[0] * d[0] + d[1] * d[1];
                        let qb = 2.0 * (ox * d[0] + oy * d[1]);
                        let qc = ox * ox + oy * oy - t.trunk_r * t.trunk_r;
                        let (ta, tb) = if qa < 1e-12 {
                            if qc <= 0.0 { (0.0, 1.0) } else { (1.0, 0.0) }
                        } else {
                            let disc = qb * qb - 4.0 * qa * qc;
                            if disc < 0.0 {
                                (1.0, 0.0)
                            } else {
                                let sq = disc.sqrt();
                                ((-qb - sq) / (2.0 * qa), (-qb + sq) / (2.0 * qa))
                            }
                        };
                        let ta = ta.max(0.0);
                        let tb = tb.min(1.0);
                        if tb >= ta {
                            let za = a[2] + d[2] * ta;
                            let zb = a[2] + d[2] * tb;
                            let (lo, hi) = (za.min(zb), za.max(zb));
                            if hi >= t.z && lo <= t.z + t.crown_base {
                                trunk_hit = true;
                            }
                        }
                    }
                }
                if trunk_hit || len > stop_len {
                    return (trunk_hit, len);
                }
            }
            if t1 >= 1.0 {
                break;
            }
            if tmx < tmy {
                i += sx;
                tmx += tdx;
                if (sx > 0 && i >= n_x) || (sx < 0 && i < 0) {
                    break;
                }
            } else {
                j += sy;
                tmy += tdy;
                if (sy > 0 && j >= n_y) || (sy < 0 && j < 0) {
                    break;
                }
            }
            t0 = t1;
        }
        (trunk_hit, len)
    }
}
