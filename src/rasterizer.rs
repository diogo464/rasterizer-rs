mod framebuffer;
mod model;
mod point;
mod shader;
mod vertex_data;

use crate::PPMImage;
use point::Point;
use shader::Interpolate;

pub use framebuffer::Framebuffer;
pub use model::{Model, ModelFace};
pub use shader::{ShaderData, ShaderField, ShaderProgram, Texture};
pub use vertex_data::VertexData;

use glm::{Vec2, Vec3, Vec4};
use nalgebra_glm as glm;
use rand::Rng;

pub struct Rasterizer {
    framebuffer: Framebuffer,
}

impl Rasterizer {
    pub fn new(width: u32, height: u32) -> Self {
        let framebuffer = Framebuffer::new(width, height);
        Self { framebuffer }
    }

    pub fn render_model(&mut self, model: &Model, program: &ShaderProgram, uniform: &ShaderData) {
        for face in model.faces.iter() {
            let (v0, v1, v2) = model.get_face_vertices(face);
            draw_triangle(&mut self.framebuffer, program, uniform, v0, v1, v2);
        }
    }

    pub fn framebuffer(&self) -> &Framebuffer {
        &self.framebuffer
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

fn draw_line(fb: &mut Framebuffer, program: &ShaderProgram, start: &Vec3, end: &Vec3) {
    let (x0, y0) = normalized_to_screen(start.x, start.y, fb.width(), fb.height());
    let (x1, y1) = normalized_to_screen(end.x, end.y, fb.width(), fb.height());
    let (mut x0, mut y0) = (x0 as i32, y0 as i32);
    let (x1, y1) = (x1 as i32, y1 as i32);

    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut e2; /* error value e_xy */

    let line_length = dx + dy;

    loop {
        let current_distance = x0 + y0;
        let current_fraction = current_distance as f32 / line_length as f32;
        let current_depth = nalgebra_glm::lerp_scalar(start.z, end.z, current_fraction);
        if fb.set_depth_if_greater(x0 as u32, y0 as u32, current_depth) {
            fb.set_color(x0 as u32, y0 as u32, &Vec3::new(1.0, 1.0, 1.0));
        }

        if x0 == x1 && y0 == y1 {
            break;
        }
        e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        } /* e_xy+e_x > 0 */
        if e2 <= dx {
            err += dx;
            y0 += sy;
        } /* e_xy+e_y < 0 */
    }
}

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

fn draw_triangle(
    fb: &mut Framebuffer,
    program: &ShaderProgram,
    uniform: &ShaderData,
    v0: &VertexData,
    v1: &VertexData,
    v2: &VertexData,
) {
    let (v0_pos, v0_data) = program.run_vertex(uniform, v0);
    let (v1_pos, v1_data) = program.run_vertex(uniform, v1);
    let (v2_pos, v2_data) = program.run_vertex(uniform, v2);

    let inrange = |l, h, v| v >= l && v <= h;

    if !inrange(-1.0, 1.0, v0_pos.z)
        && !inrange(-1.0, 1.0, v1_pos.z)
        && !inrange(-1.0, 1.0, v2_pos.z)
    {
        return;
    }

    // if !(v0_pos.x >= -1.0 && v0_pos.x <= 1.0 && v0_pos.y >= -1.0 && v0_pos.y <= 1.0) &&
    // if !(v1_pos.x >= -1.0 && v1_pos.x <= 1.0 && v1_pos.y >= -1.0 && v1_pos.y <= 1.0) &&
    // !(v2_pos.x >= -1.0 && v2_pos.x <= 1.0 && v2_pos.y >= -1.0 && v2_pos.y <= 1.0){}

    let col1 = v1_pos - v0_pos;
    let col2 = v2_pos - v0_pos;
    let inv_det = 1.0 / (col1.x * col2.y - col2.x * col1.y);
    let inv_col1 = Vec2::new(col2.y, -col1.y) * inv_det;
    let inv_col2 = Vec2::new(-col2.x, col1.x) * inv_det;

    let p0 = Point::from_normalized(v0_pos.x, v0_pos.y, fb.width(), fb.height());
    let p1 = Point::from_normalized(v1_pos.x, v1_pos.y, fb.width(), fb.height());
    let p2 = Point::from_normalized(v2_pos.x, v2_pos.y, fb.width(), fb.height());

    let (tl, br) = get_triangle_bounding_box(&p0, &p1, &p2);

    for y in tl.y.clamp(0, fb.height() as i32 - 1)..=br.y.clamp(0, fb.height() as i32 - 1) {
        for x in tl.x.clamp(0, fb.width() as i32 - 1)..=br.x.clamp(0, fb.width() as i32 - 1) {
            let (target_x, target_y) = {
                let (nx, ny) = screen_to_normalized(x as u32, y as u32, fb.width(), fb.height());
                (nx - v0_pos.x, ny - v0_pos.y)
            };
            let newbase_x = inv_col1.x * target_x + inv_col2.x * target_y;
            let newbase_y = inv_col1.y * target_x + inv_col2.y * target_y;
            let local_coords = Vec2::new(newbase_x, newbase_y);

            if local_coords.x >= 0.0
                && local_coords.y >= 0.0
                && (local_coords.x + local_coords.y) <= 1.0
            {
                let ratio_2 = newbase_x;
                let ratio_1 = newbase_y;
                let ratio_0 = 1.0 - ratio_1 - ratio_2;
                let fragment_depth = ratio_0 * v0_pos.z + ratio_1 * v1_pos.z + ratio_2 * v2_pos.z;

                if
                //inrange(0.0, 1.0, fragment_depth)
                fb.set_depth_if_greater(x as u32, y as u32, fragment_depth) {
                    let fragment_data = ShaderData::interpolate(
                        &v0_data, &v1_data, &v2_data, ratio_0, ratio_1, ratio_2,
                    );
                    let color = program.run_fragment(uniform, &fragment_data);
                    fb.set_color(x as u32, y as u32, &color.xyz());
                }
            }
        }
    }
}

pub fn test() {
    const WIDTH: u32 = 400;
    const HEIGHT: u32 = 400;

    let mut image = PPMImage::new(WIDTH * 2, HEIGHT);
    let mut framebuffer = Framebuffer::new(WIDTH, HEIGHT);

    let program = ShaderProgram::new(
        |_, v, output| {
            let color = match v.position.z {
                0.0 => Vec4::new(1.0, 0.0, 0.0, 0.0),
                1.0 => Vec4::new(0.0, 1.0, 0.0, 0.0),
                2.0 => Vec4::new(0.0, 0.0, 1.0, 0.0),
                _ => Vec4::new(1.0, 1.0, 1.0, 1.0),
            };

            output.set("color", ShaderField::Vector4(color));
            let mut rng = rand::thread_rng();
            Vec4::new(
                v.position.x, // + rng.gen_range(-0.1, 0.1),
                v.position.y, // + rng.gen_range(-0.1, 0.1),
                v.position.z, // + rng.gen_range(-0.1, 0.1),
                1.0,
            )
        },
        |_, input| {
            if let ShaderField::Vector4(c) = input.get("color").unwrap() {
                *c
            } else {
                Vec4::new(1.0, 1.0, 1.0, 1.0)
            }
        },
    );
    let uniform = ShaderData::new();
    let v0 = VertexData::from_position(Vec3::new(-0.5, 0.5, 1.0));
    let v1 = VertexData::from_position(Vec3::new(0.5, 0.5, 1.0));
    let v2 = VertexData::from_position(Vec3::new(0.0, -0.5, 1.0));
    draw_triangle(&mut framebuffer, &program, &uniform, &v0, &v1, &v2);

    let v0 = VertexData::from_position(Vec3::new(-0.2, 0.5, 0.5));
    let v1 = VertexData::from_position(Vec3::new(0.9, 0.5, 0.5));
    let v2 = VertexData::from_position(Vec3::new(0.0, -0.5, 0.5));
    draw_triangle(&mut framebuffer, &program, &uniform, &v0, &v1, &v2);

    for (x, y) in image.pixels() {
        if x >= WIDTH {
            let pixel = framebuffer.get_color(x - WIDTH, y).unwrap();
            image.set_pixel(x, y, pixel.x, pixel.y, pixel.z);
        } else {
            let depth = framebuffer.get_depth(x, y).unwrap().max(0.0).min(1.0);
            image.set_pixel(x, y, depth, depth, depth);
        }
    }

    image
        .write_to(std::fs::File::create("raster.ppm").unwrap())
        .unwrap();
}
