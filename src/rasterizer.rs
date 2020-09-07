mod framebuffer;
mod model;
mod point;
mod shader;
mod vertex_data;

use crate::PPMImage;
use point::Point;
use rayon::prelude::*;
use shader::{Interpolate, ShaderDataIterpolator};

pub use framebuffer::Framebuffer;
pub use model::{Model, ModelFace};
pub use shader::{ShaderData, ShaderField, ShaderProgram, Texture};
pub use vertex_data::VertexData;

use glm::{Vec2, Vec3, Vec4};
use nalgebra_glm as glm;
use rand::Rng;

#[derive(Debug, Copy, Clone, PartialEq)]
struct BoundingBox {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl BoundingBox {
    fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    fn x(&self) -> u32 {
        self.x
    }
    fn y(&self) -> u32 {
        self.y
    }
    fn width(&self) -> u32 {
        self.w
    }
    fn height(&self) -> u32 {
        self.h
    }
    fn overlap(&self, other: &BoundingBox) -> Option<BoundingBox> {
        let (minl, maxl) = if self.x <= other.x {
            (self.x, other.x)
        } else {
            (other.x, self.x)
        };
        let (minr, maxr) = if (self.x + self.w) <= (other.x + other.w) {
            (self.x + self.w, other.x + other.w)
        } else {
            (other.x + other.w, self.x + self.w)
        };

        if minr < maxl {
            return None;
        }

        let (mint, maxt) = if self.y <= other.y {
            (self.y, other.y)
        } else {
            (other.y, self.y)
        };
        let (minb, maxb) = if (self.y + self.h) <= (other.y + other.h) {
            (self.y + self.h, other.y + other.h)
        } else {
            (other.y + other.h, self.y + self.h)
        };

        if maxt > minb {
            return None;
        }

        let width = minr - maxl;
        let height = minb - maxt;
        Some(BoundingBox::new(maxl, maxt, width, height))
    }
}

#[derive(Debug, Clone)]
pub struct Fragment {
    pub color: Vec3,
    pub depth: f32,
    //Index to ProcessedModelFace
    face: usize,
    vertex0_ratio: f32,
    vertex1_ratio: f32,
    vertex2_ratio: f32,
}

impl Fragment {
    const INVALID_FACE_INDEX: usize = usize::MAX;
}

impl Default for Fragment {
    fn default() -> Self {
        Self {
            color: Vec3::new(1.0, 1.0, 1.0),
            depth: f32::MAX,
            face: Self::INVALID_FACE_INDEX,
            vertex0_ratio: 0.0,
            vertex1_ratio: 0.0,
            vertex2_ratio: 0.0,
        }
    }
}

pub struct FragmentBuffer {
    width: u32,
    height: u32,
    pub fragments: Vec<Fragment>,
}

impl FragmentBuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        let fragments = vec![Fragment::default(); size];
        Self {
            width,
            height,
            fragments,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn get_color(&self, x: u32, y: u32) -> Option<&Vec3> {
        self.fragments
            .get(self.coords_to_index(x, y) as usize)
            .map(|f| &f.color)
    }

    fn coords_to_index(&self, x: u32, y: u32) -> u32 {
        x + y * self.width
    }
}

struct FrameBlock {
    // The block on the screen this struct represents
    bounding_box: BoundingBox,
    //indices to ProcessedModelFace
    face_indices: Vec<usize>,
    fragments: Vec<Fragment>,
}

impl FrameBlock {
    fn new(bounding_box: BoundingBox) -> Self {
        let size = bounding_box.width() * bounding_box.height();
        let face_indices = Vec::with_capacity(128);
        let fragments = vec![Fragment::default(); size as usize];
        Self {
            bounding_box,
            face_indices,
            fragments,
        }
    }
}

struct TriangleInteriorChecker {
    vertex0: Vec3,
    inv_col1: Vec2,
    inv_col2: Vec2,
}

impl TriangleInteriorChecker {
    fn new(v0: &Vec3, v1: &Vec3, v2: &Vec3) -> Self {
        let col1 = v1 - v0;
        let col2 = v2 - v0;
        let inv_det = 1.0 / (col1.x * col2.y - col2.x * col1.y);
        let inv_col1 = Vec2::new(col2.y, -col1.y) * inv_det;
        let inv_col2 = Vec2::new(-col2.x, col1.x) * inv_det;
        Self {
            vertex0: *v0,
            inv_col1,
            inv_col2,
        }
    }

    fn to_triangle_coords(&self, point: &Vec2) -> Vec2 {
        let (target_x, target_y) = { (point.x - self.vertex0.x, point.y - self.vertex0.y) };
        let newbase_x = self.inv_col1.x * target_x + self.inv_col2.x * target_y;
        let newbase_y = self.inv_col1.y * target_x + self.inv_col2.y * target_y;
        Vec2::new(newbase_x, newbase_y)
    }

    //A point in triangle coords
    fn is_point_in_triangle(&self, triangle_point: &Vec2) -> bool {
        0.0 <= triangle_point.x
            && 0.0 <= triangle_point.y
            && (triangle_point.x + triangle_point.y) <= 1.0
    }
}

struct ProcessedModelFace {
    vertex0: Vec3,
    vertex0_data: ShaderData,
    vertex1: Vec3,
    vertex1_data: ShaderData,
    vertex2: Vec3,
    vertex2_data: ShaderData,
    bounding_box: BoundingBox,
}

pub struct Rasterizer {
    framebuffer: Framebuffer,
    fragmentbuffer: Option<FragmentBuffer>,
    processed_faces: Option<Vec<ProcessedModelFace>>,
    frame_blocks: Option<Vec<FrameBlock>>,
}

impl Rasterizer {
    const NORMALIZED_COORDS_MIN: f32 = -1.0;
    const NORMALIZED_COORDS_MAX: f32 = 1.0;
    const BLOCK_SIZE: u32 = 64;

    pub fn new(width: u32, height: u32) -> Self {
        let framebuffer = Framebuffer::new(width, height);
        let fragmentbuffer = FragmentBuffer::new(width, height);
        let processed_faces = Vec::with_capacity(32000);
        let mut frame_blocks = Vec::new();

        for y in (0..height).step_by(Self::BLOCK_SIZE as usize) {
            for x in (0..width).step_by(Self::BLOCK_SIZE as usize) {
                let w = Self::BLOCK_SIZE.min(width - x);
                let h = Self::BLOCK_SIZE.min(height - y);
                let bb = BoundingBox::new(x, y, w, h);
                frame_blocks.push(FrameBlock::new(bb));
            }
        }

        Self {
            framebuffer,
            fragmentbuffer: Some(fragmentbuffer),
            processed_faces: Some(processed_faces),
            frame_blocks: Some(frame_blocks),
        }
    }

    pub fn render_model(&mut self, model: &Model, program: &ShaderProgram, uniform: &ShaderData) {
        //Need to do this because of ownership
        let mut processed_faces = self.processed_faces.take().unwrap();
        let mut fragment_buffer = self.fragmentbuffer.take().unwrap();
        let mut frame_blocks = self.frame_blocks.take().unwrap();

        let start = std::time::Instant::now();
        //Vertex shader stage
        model
            .faces
            .par_iter()
            .enumerate()
            .map(|(face_index, face)| {
                let (v0, v1, v2) = model.get_face_vertices(face);
                let (vertex0, vertex0_data) = program.run_vertex(uniform, v0);
                let (vertex1, vertex1_data) = program.run_vertex(uniform, v1);
                let (vertex2, vertex2_data) = program.run_vertex(uniform, v2);
                let bb = self.bounding_box_from_vertices(&vertex0, &vertex1, &vertex2);
                ProcessedModelFace {
                    vertex0,
                    vertex0_data,
                    vertex1,
                    vertex1_data,
                    vertex2,
                    vertex2_data,
                    bounding_box: bb,
                }
            })
            .collect_into_vec(&mut processed_faces);

        processed_faces.retain(|f| self.is_face_in_screen(f));
        //Put all the faces in their respective blocks so we can multi thread this
        let blocks_width =
            self.width() / Self::BLOCK_SIZE + (self.width() % Self::BLOCK_SIZE).min(1);
        for (face_index, face) in processed_faces.iter().enumerate() {
            if !self.is_face_in_screen(face) {
                continue;
            }
            let bb = &face.bounding_box;
            let left_block = bb.x() / Self::BLOCK_SIZE;
            let right_block = (bb.x() + bb.width()).min(self.width()) / Self::BLOCK_SIZE;
            let top_block = bb.y() / Self::BLOCK_SIZE;
            let bot_block = (bb.y() + bb.height()).min(self.height()) / Self::BLOCK_SIZE;

            for x in left_block..=right_block {
                for y in top_block..=bot_block {
                    let block_index = (x + y * blocks_width) as usize;
                    frame_blocks[block_index].face_indices.push(face_index);
                }
            }
        }

        let delta = start.elapsed().as_millis();
        println!("Vertex shader stage : {}ms", delta);

        let start = std::time::Instant::now();
        //Rasterization stage
        // for (face_index, face) in processed_faces.iter().enumerate() {
        //     Self::fill_triangle(&mut fragment_buffer, face, face_index);
        // }
        frame_blocks.par_iter_mut().for_each(|block| {
            for (face_index, face) in block
                .face_indices
                .iter()
                .map(|index| (*index, &processed_faces[*index]))
            {
                //If they didnt overlap we wouldnt be rasterizing this face here
                let rasterize_box = block.bounding_box.overlap(&face.bounding_box).unwrap();
                let triangle_checker =
                    TriangleInteriorChecker::new(&face.vertex0, &face.vertex1, &face.vertex2);

                //println!("Rasterize box : {:?}", rasterize_box);
                for y in rasterize_box.y()..(rasterize_box.y() + rasterize_box.height()) {
                    for x in rasterize_box.x()..(rasterize_box.x() + rasterize_box.width()) {
                        let (nx, ny) = screen_to_normalized(x, y, self.width(), self.height());
                        let triangle_point =
                            triangle_checker.to_triangle_coords(&Vec2::new(nx, ny));
                        if triangle_checker.is_point_in_triangle(&triangle_point) {
                            let ratio_2 = triangle_point.x;
                            let ratio_1 = triangle_point.y;
                            let ratio_0 = 1.0 - ratio_1 - ratio_2;
                            let fragment_depth = ratio_0 * face.vertex0.z
                                + ratio_1 * face.vertex1.z
                                + ratio_2 * face.vertex2.z;
                            let fragment_index = {
                                let fragment_x = x - block.bounding_box.x();
                                let fragment_y = y - block.bounding_box.y();
                                (fragment_x + fragment_y * block.bounding_box.width()) as usize
                            };

                            let fragment = &mut block.fragments[fragment_index];
                            if fragment.depth > fragment_depth {
                                fragment.depth = fragment_depth;
                                fragment.face = face_index;
                                fragment.vertex0_ratio = ratio_0;
                                fragment.vertex1_ratio = ratio_1;
                                fragment.vertex2_ratio = ratio_2;
                            }
                        }
                    }
                }
            }
        });

        let delta = start.elapsed().as_millis();
        println!("Rasterization stage : {}ms", delta);
        let start = std::time::Instant::now();
        //Fragment shader stage
        let width = self.width() as usize;
        self.framebuffer
            .color
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, color)| {
                assert!(index < 1920 * 1080);
                let x = index % width;
                let y = index / width;
                let block_index = x / Self::BLOCK_SIZE as usize
                    + (y / Self::BLOCK_SIZE as usize) * blocks_width as usize;
                //println!("X : {} , Y : {}, index : {}, height : {}", x, y, block_index, height);
                let block = &frame_blocks[block_index];
                let fragment_x = x - block.bounding_box.x() as usize;
                let fragment_y = y - block.bounding_box.y() as usize;
                let fragment_index = fragment_x + fragment_y * block.bounding_box.width() as usize;
                let fragment = &block.fragments[fragment_index];
                if fragment.face != Fragment::INVALID_FACE_INDEX {
                    let face = &processed_faces[fragment.face];
                    let interpolator = ShaderDataIterpolator {
                        vertex0_data: &face.vertex0_data,
                        vertex1_data: &face.vertex1_data,
                        vertex2_data: &face.vertex2_data,
                        vertex0_ratio: fragment.vertex0_ratio,
                        vertex1_ratio: fragment.vertex1_ratio,
                        vertex2_ratio: fragment.vertex2_ratio,
                    };
                    *color = program.run_fragment(uniform, &interpolator).xyz();
                }
            });
        // fragment_buffer.fragments.par_iter_mut().for_each(|frag| {
        //     if frag.face == Fragment::INVALID_FACE_INDEX {
        //         return;
        //     }

        //     let face = &processed_faces[frag.face];
        //     let interpolator = ShaderDataIterpolator {
        //         vertex0_data: &face.vertex0_data,
        //         vertex1_data: &face.vertex1_data,
        //         vertex2_data: &face.vertex2_data,
        //         vertex0_ratio: frag.vertex0_ratio,
        //         vertex1_ratio: frag.vertex1_ratio,
        //         vertex2_ratio: frag.vertex2_ratio,
        //     };
        //     let color = program.run_fragment(uniform, &interpolator);
        //     frag.color = color.xyz();
        // });

        let delta = start.elapsed().as_millis();
        println!("Shading stage : {}ms", delta);

        self.fragmentbuffer = Some(fragment_buffer);
        self.processed_faces = Some(processed_faces);
        // for face in model.faces.iter() {
        //     let (v0, v1, v2) = model.get_face_vertices(face);
        //     draw_triangle(&mut self.framebuffer, program, uniform, v0, v1, v2);
        // }
    }

    pub fn width(&self) -> u32 {
        self.framebuffer.width()
    }

    pub fn height(&self) -> u32 {
        self.framebuffer.height()
    }

    pub fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
    }

    pub fn fragment_buffer(&self) -> &FragmentBuffer {
        self.fragmentbuffer.as_ref().unwrap()
    }

    fn is_face_in_screen(&self, face: &ProcessedModelFace) -> bool {
        self.is_point_in_screen(&face.vertex0)
            && self.is_point_in_screen(&face.vertex1)
            && self.is_point_in_screen(&face.vertex2)
    }

    fn fill_triangle(
        framebuffer: &mut FragmentBuffer,
        face: &ProcessedModelFace,
        face_index: usize,
    ) {
        // let col1 = face.vertex1 - face.vertex0;
        // let col2 = face.vertex2 - face.vertex0;
        // let inv_det = 1.0 / (col1.x * col2.y - col2.x * col1.y);
        // let inv_col1 = Vec2::new(col2.y, -col1.y) * inv_det;
        // let inv_col2 = Vec2::new(-col2.x, col1.x) * inv_det;
        let triangle_checker =
            TriangleInteriorChecker::new(&face.vertex0, &face.vertex1, &face.vertex2);

        let clamped_point_from_normalized = |point: &Vec3| -> Point {
            let mut p =
                Point::from_normalized(point.x, point.y, framebuffer.width(), framebuffer.height());
            p.x = p.x.clamp(0, framebuffer.width() as i32 - 1);
            p.y = p.y.clamp(0, framebuffer.height() as i32 - 1);
            p
        };

        let p0 = clamped_point_from_normalized(&face.vertex0);
        let p1 = clamped_point_from_normalized(&face.vertex1);
        let p2 = clamped_point_from_normalized(&face.vertex2);

        let (tl, br) = get_triangle_bounding_box(&p0, &p1, &p2);

        for y in tl.y..=br.y {
            for x in tl.x..=br.x {
                // let (target_x, target_y) = {
                //     let (nx, ny) = screen_to_normalized(
                //         x as u32,
                //         y as u32,
                //         framebuffer.width(),
                //         framebuffer.height(),
                //     );
                //     (nx - face.vertex0.x, ny - face.vertex0.y)
                // };
                // let newbase_x = inv_col1.x * target_x + inv_col2.x * target_y;
                // let newbase_y = inv_col1.y * target_x + inv_col2.y * target_y;
                // let local_coords = Vec2::new(newbase_x, newbase_y);

                let (px, py) = screen_to_normalized(
                    x as u32,
                    y as u32,
                    framebuffer.width(),
                    framebuffer.height(),
                );
                let triangle_point = triangle_checker.to_triangle_coords(&Vec2::new(px, py));

                // if local_coords.x >= 0.0
                //     && local_coords.y >= 0.0
                //     && (local_coords.x + local_coords.y) <= 1.0
                // {
                if triangle_checker.is_point_in_triangle(&triangle_point) {
                    let ratio_2 = triangle_point.x;
                    let ratio_1 = triangle_point.y;
                    // let ratio_2 = newbase_x;
                    // let ratio_1 = newbase_y;

                    let ratio_0 = 1.0 - ratio_1 - ratio_2;
                    let fragment_depth = ratio_0 * face.vertex0.z
                        + ratio_1 * face.vertex1.z
                        + ratio_2 * face.vertex2.z;

                    let fragment_index = (x as u32 + y as u32 * framebuffer.width()) as usize;
                    let fragment = &mut framebuffer.fragments[fragment_index];
                    if fragment.depth > fragment_depth {
                        fragment.depth = fragment_depth;
                        fragment.face = face_index;
                        fragment.vertex0_ratio = ratio_0;
                        fragment.vertex1_ratio = ratio_1;
                        fragment.vertex2_ratio = ratio_2;
                    }

                    // if framebuffer.set_depth_if_greater(x as u32, y as u32, fragment_depth) {
                    //     // let fragment_data = ShaderData::interpolate(
                    //     //     &face.vertex0_data,
                    //     //     &face.vertex1_data,
                    //     //     &face.vertex2_data,
                    //     //     ratio_0,
                    //     //     ratio_1,
                    //     //     ratio_2,
                    //     // );
                    //     let interpolator = ShaderDataIterpolator {
                    //         vertex0_data: &face.vertex0_data,
                    //         vertex1_data: &face.vertex1_data,
                    //         vertex2_data: &face.vertex2_data,
                    //         vertex0_ratio: ratio_0,
                    //         vertex1_ratio: ratio_1,
                    //         vertex2_ratio: ratio_2,
                    //     };
                    //     let color = program.run_fragment(uniform, &interpolator);
                    //     framebuffer.set_color(x as u32, y as u32, &color.xyz());
                    // }
                }
            }
        }
    }

    //Checks if a point in normalized coordinates is inside the screen
    fn is_point_in_screen(&self, point: &Vec3) -> bool {
        let inrange = |l, h, v| v >= l && v <= h;
        inrange(
            Self::NORMALIZED_COORDS_MIN,
            Self::NORMALIZED_COORDS_MAX,
            point.x,
        ) && inrange(
            Self::NORMALIZED_COORDS_MIN,
            Self::NORMALIZED_COORDS_MAX,
            point.y,
        ) && inrange(
            Self::NORMALIZED_COORDS_MIN,
            Self::NORMALIZED_COORDS_MAX,
            point.z,
        )
    }

    fn bounding_box_from_vertices(&self, v0: &Vec3, v1: &Vec3, v2: &Vec3) -> BoundingBox {
        let min_x = v0.x.min(v1.x.min(v2.x));
        let min_y = v0.y.min(v1.y.min(v2.y));
        let max_x = v0.x.max(v1.x.max(v2.x));
        let max_y = v0.y.max(v1.y.max(v2.y));

        let (tlx, tly) = normalized_to_screen(min_x, max_y, self.width(), self.height());
        let (brx, bry) = normalized_to_screen(max_x, min_y, self.width(), self.height());

        //we add 1 to round it up so the box completly fills the triangle otherwise some triangles will render weirdly
        BoundingBox::new(tlx, tly, brx - tlx + 1, bry - tly + 1)
    }
}

fn normalized_to_screen(mut x: f32, mut y: f32, width: u32, height: u32) -> (u32, u32) {
    x = (x + 1.0) / 2.0;
    y = (-y + 1.0) / 2.0;
    let screen_x = ((x * width as f32) as u32).min(width - 1);
    let screen_y = ((y * height as f32) as u32).min(height - 1);
    (screen_x, screen_y)
}

fn screen_to_normalized(x: u32, y: u32, width: u32, height: u32) -> (f32, f32) {
    let normalized_x = (x as f32 / width as f32) * 2.0 - 1.0;
    let normalized_y = -((y as f32 / height as f32) * 2.0 - 1.0);
    (normalized_x, normalized_y)
}

// fn draw_line(fb: &mut Framebuffer, program: &ShaderProgram, start: &Vec3, end: &Vec3) {
//     let (x0, y0) = normalized_to_screen(start.x, start.y, fb.width(), fb.height());
//     let (x1, y1) = normalized_to_screen(end.x, end.y, fb.width(), fb.height());
//     let (mut x0, mut y0) = (x0 as i32, y0 as i32);
//     let (x1, y1) = (x1 as i32, y1 as i32);

//     let dx = (x1 - x0).abs();
//     let sx = if x0 < x1 { 1 } else { -1 };
//     let dy = -(y1 - y0).abs();
//     let sy = if y0 < y1 { 1 } else { -1 };
//     let mut err = dx + dy;
//     let mut e2; /* error value e_xy */
//     let line_length = dx + dy;

//     loop {
//         let current_distance = x0 + y0;
//         let current_fraction = current_distance as f32 / line_length as f32;
//         let current_depth = nalgebra_glm::lerp_scalar(start.z, end.z, current_fraction);
//         if fb.set_depth_if_greater(x0 as u32, y0 as u32, current_depth) {
//             fb.set_color(x0 as u32, y0 as u32, &Vec3::new(1.0, 1.0, 1.0));
//         }

//         if x0 == x1 && y0 == y1 {
//             break;
//         }
//         e2 = 2 * err;
//         if e2 >= dy {
//             err += dy;
//             x0 += sx;
//         } /* e_xy+e_x > 0 */
//         if e2 <= dx {
//             err += dx;
//             y0 += sy;
//         } /* e_xy+e_y < 0 */
//     }
// }

fn get_triangle_bounding_box(p0: &Point, p1: &Point, p2: &Point) -> (Point, Point) {
    let tlx = std::cmp::min(p0.x, std::cmp::min(p1.x, p2.x));
    let tly = std::cmp::min(p0.y, std::cmp::min(p1.y, p2.y));
    let brx = std::cmp::max(p0.x, std::cmp::max(p1.x, p2.x));
    let bry = std::cmp::max(p0.y, std::cmp::max(p1.y, p2.y));
    (
        Point::new(tlx as i32, tly as i32),
        Point::new(brx as i32, bry as i32),
    )
}

// fn draw_triangle(
//     fb: &mut Framebuffer,
//     program: &ShaderProgram,
//     uniform: &ShaderData,
//     v0: &VertexData,
//     v1: &VertexData,
//     v2: &VertexData,
// ) {
//     let (v0_pos, v0_data) = program.run_vertex(uniform, v0);
//     let (v1_pos, v1_data) = program.run_vertex(uniform, v1);
//     let (v2_pos, v2_data) = program.run_vertex(uniform, v2);

//     let inrange = |l, h, v| v >= l && v <= h;

//     if !inrange(-1.0, 1.0, v0_pos.z)
//         && !inrange(-1.0, 1.0, v1_pos.z)
//         && !inrange(-1.0, 1.0, v2_pos.z)
//     {
//         return;
//     }

//     // if !(v0_pos.x >= -1.0 && v0_pos.x <= 1.0 && v0_pos.y >= -1.0 && v0_pos.y <= 1.0) &&
//     // if !(v1_pos.x >= -1.0 && v1_pos.x <= 1.0 && v1_pos.y >= -1.0 && v1_pos.y <= 1.0) &&
//     // !(v2_pos.x >= -1.0 && v2_pos.x <= 1.0 && v2_pos.y >= -1.0 && v2_pos.y <= 1.0){}

//     let col1 = v1_pos - v0_pos;
//     let col2 = v2_pos - v0_pos;
//     let inv_det = 1.0 / (col1.x * col2.y - col2.x * col1.y);
//     let inv_col1 = Vec2::new(col2.y, -col1.y) * inv_det;
//     let inv_col2 = Vec2::new(-col2.x, col1.x) * inv_det;

//     let p0 = Point::from_normalized(v0_pos.x, v0_pos.y, fb.width(), fb.height());
//     let p1 = Point::from_normalized(v1_pos.x, v1_pos.y, fb.width(), fb.height());
//     let p2 = Point::from_normalized(v2_pos.x, v2_pos.y, fb.width(), fb.height());

//     let (tl, br) = get_triangle_bounding_box(&p0, &p1, &p2);

//     for y in tl.y.clamp(0, fb.height() as i32 - 1)..=br.y.clamp(0, fb.height() as i32 - 1) {
//         for x in tl.x.clamp(0, fb.width() as i32 - 1)..=br.x.clamp(0, fb.width() as i32 - 1) {
//             let (target_x, target_y) = {
//                 let (nx, ny) = screen_to_normalized(x as u32, y as u32, fb.width(), fb.height());
//                 (nx - v0_pos.x, ny - v0_pos.y)
//             };
//             let newbase_x = inv_col1.x * target_x + inv_col2.x * target_y;
//             let newbase_y = inv_col1.y * target_x + inv_col2.y * target_y;
//             let local_coords = Vec2::new(newbase_x, newbase_y);

//             if local_coords.x >= 0.0
//                 && local_coords.y >= 0.0
//                 && (local_coords.x + local_coords.y) <= 1.0
//             {
//                 let ratio_2 = newbase_x;
//                 let ratio_1 = newbase_y;
//                 let ratio_0 = 1.0 - ratio_1 - ratio_2;
//                 let fragment_depth = ratio_0 * v0_pos.z + ratio_1 * v1_pos.z + ratio_2 * v2_pos.z;

//                 if fb.set_depth_if_greater(x as u32, y as u32, fragment_depth) {
//                     //let fragment_data = ShaderData::interpolate(
//                     //    &v0_data, &v1_data, &v2_data, ratio_0, ratio_1, ratio_2,
//                     //);
//                     let color = program.run_fragment(uniform, &uniform);
//                     fb.set_color(x as u32, y as u32, &color.xyz());
//                 }
//             }
//         }
//     }
// }

// pub fn test() {
//     const WIDTH: u32 = 400;
//     const HEIGHT: u32 = 400;

//     let mut image = PPMImage::new(WIDTH * 2, HEIGHT);
//     let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);

//     let program = ShaderProgram::new(
//         |_, v, output| {
//             let color = match v.position.z {
//                 0.0 => Vec4::new(1.0, 0.0, 0.0, 0.0),
//                 1.0 => Vec4::new(0.0, 1.0, 0.0, 0.0),
//                 2.0 => Vec4::new(0.0, 0.0, 1.0, 0.0),
//                 _ => Vec4::new(1.0, 1.0, 1.0, 1.0),
//             };

//             output.set("color", ShaderField::Vector4(color));
//             let mut rng = rand::thread_rng();
//             Vec4::new(
//                 v.position.x, // + rng.gen_range(-0.1, 0.1),
//                 v.position.y, // + rng.gen_range(-0.1, 0.1),
//                 v.position.z, // + rng.gen_range(-0.1, 0.1),
//                 1.0,
//             )
//         },
//         |_, input| {
//             //if let ShaderField::Vector4(c) = input.get("color").unwrap() {
//             //    *c
//             //} else {
//             //    Vec4::new(1.0, 1.0, 1.0, 1.0)
//             //}
//             Vec4::new(1.0, 1.0, 1.0, 1.0)
//         },
//     );
//     let uniform = ShaderData::new();
//     let v0 = VertexData::from_position(Vec3::new(-0.5, 0.5, 1.0));
//     let v1 = VertexData::from_position(Vec3::new(0.5, 0.5, 1.0));
//     let v2 = VertexData::from_position(Vec3::new(0.0, -0.5, 1.0));
//     draw_triangle(&mut framebuffer, &program, &uniform, &v0, &v1, &v2);

//     let v0 = VertexData::from_position(Vec3::new(-0.2, 0.5, 0.5));
//     let v1 = VertexData::from_position(Vec3::new(0.9, 0.5, 0.5));
//     let v2 = VertexData::from_position(Vec3::new(0.0, -0.5, 0.5));
//     draw_triangle(&mut framebuffer, &program, &uniform, &v0, &v1, &v2);

//     for (x, y) in image.pixels() {
//         if x >= WIDTH {
//             let pixel = framebuffer.get_color(x - WIDTH, y).unwrap();
//             image.set_pixel(x, y, pixel.x, pixel.y, pixel.z);
//         } else {
//             let depth = framebuffer.get_depth(x, y).unwrap().max(0.0).min(1.0);
//             image.set_pixel(x, y, depth, depth, depth);
//         }
//     }

//     image
//         .write_to(std::fs::File::create("raster.ppm").unwrap())
//         .unwrap();
// }
