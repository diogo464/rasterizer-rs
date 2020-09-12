mod bounding_box;
mod framebuffer;
mod frametime;
mod point;
mod shader;

use bounding_box::BoundingBox;

pub use framebuffer::Framebuffer;
pub use frametime::FrameTime;
pub use shader::{FragmentShader, Interpolate, Shader, ShaderData, VertexShader};
//pub use vertex_data::VertexData;

use glm::{Vec2, Vec3};
use itertools::Itertools;
use nalgebra_glm as glm;
use rayon::prelude::*;

#[derive(Debug, Clone)]
struct Fragment {
    color: Vec3,
    depth: f32,
    //Index to ProcessedModelFace
    face: usize,
    vertex0_ratio: f32,
    vertex1_ratio: f32,
    vertex2_ratio: f32,
}

impl Fragment {
    const INVALID_FACE_INDEX: usize = usize::MAX;

    fn is_valid(&self) -> bool {
        self.face != Self::INVALID_FACE_INDEX
    }
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

    fn clear(&mut self) {
        self.face_indices.clear();
        self.fragments.iter_mut().for_each(|f| {
            f.depth = f32::MAX;
            f.face = Fragment::INVALID_FACE_INDEX;
        });
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

struct ProcessedModelFace<DataType> {
    vertex0: Vec3,
    vertex0_data: DataType,
    vertex1: Vec3,
    vertex1_data: DataType,
    vertex2: Vec3,
    vertex2_data: DataType,
    bounding_box: BoundingBox,
}

pub struct Rasterizer {
    framebuffer: Framebuffer,
    frame_blocks: Option<Vec<FrameBlock>>,
    frame_time: FrameTime,
    frame_block_count: usize,
}

impl Rasterizer {
    const NORMALIZED_COORDS_MIN: f32 = -1.0;
    const NORMALIZED_COORDS_MAX: f32 = 1.0;
    const BLOCK_SIZE: u32 = 64;

    pub fn new(width: u32, height: u32) -> Self {
        let framebuffer = Framebuffer::new(width, height);
        let mut frame_blocks = Vec::new();

        for y in (0..height).step_by(Self::BLOCK_SIZE as usize) {
            for x in (0..width).step_by(Self::BLOCK_SIZE as usize) {
                let w = Self::BLOCK_SIZE.min(width - x);
                let h = Self::BLOCK_SIZE.min(height - y);
                let bb = BoundingBox::new(x, y, w, h);
                frame_blocks.push(FrameBlock::new(bb));
            }
        }
        let block_count = frame_blocks.len();
        Self {
            framebuffer,
            frame_blocks: Some(frame_blocks),
            frame_time: FrameTime::zero(),
            frame_block_count: block_count,
        }
    }

    pub fn render_model<VS, FS, SD, V, U>(
        &mut self,
        vertices: &[V],
        indices: &[usize],
        vertex_shader: &VS,
        fragment_shader: &FS,
        uniform: &U,
    ) where
        V: Send + Sync,
        U: Send + Sync,
        SD: ShaderData,
        VS: VertexShader<VertexData = V, Uniform = U, SharedData = SD>,
        FS: FragmentShader<Uniform = U, SharedData = SD>,
    {
        //Need to do this because of ownership
        let mut frame_blocks = self.frame_blocks.take().unwrap();
        let mut processed_faces = Vec::with_capacity(16000);

        let start = std::time::Instant::now();
        //Vertex shader stage
        (0..indices.len() / 3)
            .into_par_iter()
            .map(|triangle_index| {
                let base_index = triangle_index * 3;
                let (v0, v1, v2) = (
                    &vertices[base_index],
                    &vertices[base_index + 1],
                    &vertices[base_index + 2],
                );
                let (vertex0, vertex0_data) = vertex_shader.vertex(v0, &uniform);
                let (vertex1, vertex1_data) = vertex_shader.vertex(v1, &uniform);
                let (vertex2, vertex2_data) = vertex_shader.vertex(v2, &uniform);
                let vertex0 = vertex0.xyz() / vertex0[3];
                let vertex1 = vertex1.xyz() / vertex1[3];
                let vertex2 = vertex2.xyz() / vertex2[3];
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

        //Put all the faces in their respective blocks so we can multi thread this
        for (face_index, face) in processed_faces.iter().enumerate() {
            if self.is_face_in_screen(face) {
                let bb = &face.bounding_box;
                for block_index in self.frame_blocks_in_bounding_box(bb) {
                    frame_blocks[block_index].face_indices.push(face_index);
                }
            }
        }

        let vertex_shader_duration = start.elapsed();
        let start = std::time::Instant::now();

        frame_blocks.par_iter_mut().for_each(|block| {
            for (face_index, face) in block
                .face_indices
                .iter()
                .map(|index| (*index, &processed_faces[*index]))
            {
                if let Some(rasterize_box) = block.bounding_box.overlap(&face.bounding_box) {
                    let triangle_checker =
                        TriangleInteriorChecker::new(&face.vertex0, &face.vertex1, &face.vertex2);

                    let y_iter = rasterize_box.y()..(rasterize_box.y() + rasterize_box.height());
                    let x_iter = rasterize_box.x()..(rasterize_box.x() + rasterize_box.width());

                    for (y, x) in y_iter.cartesian_product(x_iter) {
                        let (nx, ny) = screen_to_normalized(x, y, self.width(), self.height());
                        let triangle_point =
                            triangle_checker.to_triangle_coords(&Vec2::new(nx, ny));
                        if triangle_checker.is_point_in_triangle(&triangle_point) {
                            let ratio_2 = triangle_point.y;
                            let ratio_1 = triangle_point.x;
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

        let rasterization_duration = start.elapsed();
        let start = std::time::Instant::now();

        //Fragment shader stage
        let width = self.width() as usize;
        let blocks_width = self.frame_blocks_width();
        self.framebuffer
            .color
            .par_iter_mut()
            .enumerate()
            .for_each(|(index, color)| {
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
                if fragment.is_valid() {
                    let face = &processed_faces[fragment.face];
                    let interpolated = SD::interpolate(
                        &face.vertex0_data,
                        &face.vertex1_data,
                        &face.vertex2_data,
                        fragment.vertex0_ratio,
                        fragment.vertex1_ratio,
                        fragment.vertex2_ratio,
                    );
                    *color = fragment_shader.fragment(&interpolated, &uniform).xyz();
                }
            });

        let fragment_shader_duration = start.elapsed();

        self.frame_time = FrameTime::new(
            vertex_shader_duration,
            rasterization_duration,
            fragment_shader_duration,
        );
        self.frame_blocks = Some(frame_blocks);
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

    pub fn clear(&mut self) {
        self.clear_color(Vec3::new(0.0, 0.0, 0.0))
    }

    pub fn clear_color(&mut self, color: Vec3) {
        self.frame_blocks
            .as_mut()
            .unwrap()
            .par_iter_mut()
            .for_each(|b| b.clear());
        self.framebuffer.color.fill(color);
    }

    pub fn frametime(&self) -> &FrameTime {
        &self.frame_time
    }

    fn is_face_in_screen<D>(&self, face: &ProcessedModelFace<D>) -> bool {
        self.is_point_in_screen(&face.vertex0)
            || self.is_point_in_screen(&face.vertex1)
            || self.is_point_in_screen(&face.vertex2)
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

    //Number of frameblocks per line
    fn frame_blocks_width(&self) -> u32 {
        self.width() / Self::BLOCK_SIZE + (self.width() % Self::BLOCK_SIZE).min(1)
    }

    fn frame_blocks_in_bounding_box(
        &self,
        bounding_box: &BoundingBox,
    ) -> impl Iterator<Item = usize> {
        let left_block = bounding_box.x() / Self::BLOCK_SIZE;
        let right_block =
            (bounding_box.x() + bounding_box.width()).min(self.width()) / Self::BLOCK_SIZE;
        let top_block = bounding_box.y() / Self::BLOCK_SIZE;
        let bot_block =
            (bounding_box.y() + bounding_box.height()).min(self.height()) / Self::BLOCK_SIZE;
        let blocks_width = self.frame_blocks_width();

        let block_count = self.frame_block_count;
        (top_block..=bot_block)
            .cartesian_product(left_block..=right_block)
            .map(move |(y, x)| ((x + y * blocks_width) as usize).min(block_count - 1))
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
