#![feature(test)]
#![feature(clamp)]

pub mod obj;
pub mod rasterizer;

use rasterizer::{Model, ShaderData, ShaderField, ShaderProgram, Texture};

mod ppm;
use glm::{Mat4, Vec2, Vec3, Vec4};
use image::{DynamicImage, GenericImageView};
use nalgebra_glm as glm;
use ppm::PPMImage;
use rand::Rng;

#[derive(Debug)]
struct Point {
    x: i32,
    y: i32,
}

impl Point {
    fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    fn from_normalized(mut x: f32, mut y: f32, width: u32, height: u32) -> Self {
        x = (x + 1.0) / 2.0;
        y = (-y + 1.0) / 2.0;

        let x = ((x * width as f32) as i32).min(width as i32 - 1);
        let y = ((y * width as f32) as i32).min(height as i32 - 1);

        Point::new(x, y)
    }

    fn as_vec2(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::new(0, 0)
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

trait Framebuffer {
    fn width(&self) -> u32;
    fn height(&self) -> u32;
    fn set_pixel(&mut self, x: u32, y: u32, color: &Vec3);
    fn normalized_to_screen(&self, mut x: f32, mut y: f32) -> (u32, u32) {
        normalized_to_screen(x, y, self.width(), self.height())
    }
    fn screen_to_normalized(&self, x: u32, y: u32) -> (f32, f32) {
        screen_to_normalized(x, y, self.width(), self.height())
    }
}

struct MemoryFramebuffer {
    pixels: Vec<Vec3>,
    width: u32,
    height: u32,
}

impl MemoryFramebuffer {
    fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            pixels: vec![Vec3::default(); size],
            width,
            height,
        }
    }
}

impl Framebuffer for MemoryFramebuffer {
    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn set_pixel(&mut self, x: u32, y: u32, color: &Vec3) {
        //println!("{} {}", x, y);
        let index = (x + y * self.width()) as usize;
        self.pixels[index] = *color;
    }
}

impl Framebuffer for PPMImage {
    fn width(&self) -> u32 {
        self.width()
    }
    fn height(&self) -> u32 {
        self.height()
    }
    fn set_pixel(&mut self, x: u32, y: u32, color: &Vec3) {
        self.set_pixel(x, y, color.x, color.y, color.z);
    }
}

struct DepthBuffer {
    storage: Vec<f32>,
    width: u32,
    height: u32,
}

impl DepthBuffer {
    fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        Self {
            storage: vec![f32::MAX; size],
            width,
            height,
        }
    }

    fn get_depth(&self, x: u32, y: u32) -> f32 {
        self.storage[(x + y * self.width) as usize]
    }

    fn set_depth(&mut self, x: u32, y: u32, depth: f32) {
        self.storage[(x + y * self.width) as usize] = depth;
    }

    fn is_occluded(&self, point: &Vec3) -> bool {
        let (x, y) = normalized_to_screen(point.x, point.y, self.width, self.height);
        self.get_depth(x, y) < point.z
    }
}

struct SimpleTexture {
    img: DynamicImage,
}

impl SimpleTexture {
    pub fn new(img: DynamicImage) -> Self {
        Self { img }
    }
}

impl Texture for SimpleTexture {
    fn color(&self, u: f32, v: f32) -> Vec4 {
        let x = (u * self.img.width() as f32) as u32;
        let y = ((1.0 - v) * self.img.height() as f32) as u32;
        let c = self.img.as_rgb8().unwrap();
        let p = c.get_pixel(x, y);
        Vec4::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
            1.0, //p[3] as f32 / 255.0,
        )
    }
}

fn main() {
    rasterizer::test();

    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    let mut image = PPMImage::new(WIDTH , HEIGHT);
    let model = obj::read_model("diablo3.obj");
    let img = image::open("diablo3_pose_diffuse.tga").unwrap();
    let texture = Box::new(SimpleTexture::new(img));

    let mut uniform = ShaderData::new();
    let model_mat = glm::translation(&Vec3::new(-0.1, -0.5, -0.75));
    //* glm::rotation(3.14159 / 2.0, &Vec3::new(0.0, 1.0, 0.0));
    let projection = glm::perspective(16.0 / 9.0, 3.1415 / 2.0, 0.1, 100.0);
    uniform.set("model", ShaderField::Matrix4(model_mat));
    uniform.set("projection", ShaderField::Matrix4(projection));
    uniform.set("texture", ShaderField::Texture(texture));

    let shader = ShaderProgram::new(
        |uni, v, out| {
            let model = uni.get("model").unwrap().matrix4();
            let proj = uni.get("projection").unwrap().matrix4();
            let position = Vec4::new(v.position.x, v.position.y, v.position.z, 1.0);
            out.set("tcoords", ShaderField::Vector2(v.texture.unwrap()));
            let glpos = proj * model * position;
            println!("Z : {}", glpos.z / glpos[3]);
            glpos
        },
        |uni, input| {
            let tcoords = input.get("tcoords").unwrap().vector2();
            let tex = uni.get("texture").unwrap().texture();
            tex.color(tcoords.x, tcoords.y)
            //println!("Coords : {}", tcoords);
            //Vec4::new(tcoords.x, tcoords.y, 0.3, 1.0)
        },
    );
    let mut rasterizer = rasterizer::Rasterizer::new(WIDTH, HEIGHT);
    rasterizer.render_model(&model, &shader, &uniform);

    for (x, y) in image.pixels() {
        // if x >= WIDTH {
            let pixel = rasterizer.framebuffer().get_color(x, y).unwrap();
            image.set_pixel(x, y, pixel.x, pixel.y, pixel.z);
        // } 
        // else {
        //     let depth = rasterizer
        //         .framebuffer()
        //         .get_depth(x, y)
        //         .unwrap()
        //         .max(0.0)
        //         .min(1.0);
        //     image.set_pixel(x, y, depth, depth, depth);
        // }
    }

    // let mut fb = MemoryFramebuffer::new(WIDTH, HEIGHT);
    // let mut depth_buffer = DepthBuffer::new(WIDTH, HEIGHT);
    // let mut rng = rand::thread_rng();
    // let white = Vec3::new(1.0, 1.0, 1.0);
    // let red = Vec3::new(1.0, 0.0, 0.0);

    // //draw_triangle(&mut fb, &mut depth_buffer, &Vec3::new(-0.5, 0.5, 0.0), &Vec3::new(0.5, 0.5, 0.0), &Vec3::new(0.0, -0.5, 0.0), &red);

    // let translation = glm::translation(&Vec3::new(0.0, 0.0, -1.0));
    // let rotation = glm::rotation(3.14159 / 2.0, &Vec3::new(0.0, 1.0, 0.0));
    // let projection = glm::perspective(16.0 / 9.0, 3.1415 / 2.0, 0.1, 100.0);

    // for face in model.faces.iter() {
    //     let v0 = &model.vertices[face.vertex_0];
    //     let v1 = &model.vertices[face.vertex_1];
    //     let v2 = &model.vertices[face.vertex_2];

    //     let p0 = projection
    //         * translation
    //         * rotation
    //         * Vec4::new(v0.position.x, v0.position.y, v0.position.z, 1.0);
    //     let p1 = projection
    //         * translation
    //         * rotation
    //         * Vec4::new(v1.position.x, v1.position.y, v1.position.z, 1.0);
    //     let p2 = projection
    //         * translation
    //         * rotation
    //         * Vec4::new(v2.position.x, v2.position.y, v2.position.z, 1.0);

    //     let p0 = p0.xyz() / p0.w;
    //     let p1 = p1.xyz() / p1.w;
    //     let p2 = p2.xyz() / p2.w;

    //     let color = Vec3::new(1.0, 1.0, 1.0)
    //         * ((v0.normal.unwrap() + v1.normal.unwrap() + v2.normal.unwrap()) / 3.0)
    //             .dot(&Vec3::new(0.0, 0.0, -1.0))
    //             .abs();

    //     draw_triangle(&mut fb, &mut depth_buffer, &p0, &p1, &p2, &color);
    // }

    // for (x, y) in image.pixels() {
    //     if x >= 512 {
    //         let pixel = fb.pixels[((x - 512) + y * fb.width()) as usize];
    //         image.set_pixel(x, y, pixel.x, pixel.y, pixel.z);
    //     } else {
    //         let depth = depth_buffer.get_depth(x, y).max(0.0).min(1.0);
    //         image.set_pixel(x, y, depth, depth, depth);
    //     }
    // }

    image
        .write_to(std::fs::File::create("ouput.ppm").unwrap())
        .unwrap();
}

fn draw_line<F: Framebuffer>(
    fb: &mut F,
    depth_buffer: &mut DepthBuffer,
    start: &Vec3,
    end: &Vec3,
    color: &Vec3,
) {
    let (x0, y0) = fb.normalized_to_screen(start.x, start.y);
    let (x1, y1) = fb.normalized_to_screen(end.x, end.y);
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
        if depth_buffer.get_depth(x0 as u32, y0 as u32) > current_depth {
            depth_buffer.set_depth(x0 as u32, y0 as u32, current_depth);
            fb.set_pixel(x0 as u32, y0 as u32, &color);
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

fn draw_triangle<F: Framebuffer>(
    fb: &mut F,
    depth_buffer: &mut DepthBuffer,
    v0: &Vec3,
    v1: &Vec3,
    v2: &Vec3,
    color: &Vec3,
) {
    //draw_line(fb, depth_buffer, v0, v1, color);
    //draw_line(fb, depth_buffer, v0, v2, color);
    //draw_line(fb, depth_buffer, v1, v2, color);

    // if depth_buffer.is_occluded(&v0)
    //     && depth_buffer.is_occluded(&v1)
    //     && depth_buffer.is_occluded(&v2)
    // {
    //     return;
    // }

    let col1 = v1 - v0;
    let col2 = v2 - v0;
    let inv_det = 1.0 / (col1.x * col2.y - col2.x * col1.y);
    let inv_col1 = Vec2::new(col2.y, -col1.y) * inv_det;
    let inv_col2 = Vec2::new(-col2.x, col1.x) * inv_det;

    let p0 = Point::from_normalized(v0.x, v0.y, fb.width(), fb.height());
    let p1 = Point::from_normalized(v1.x, v1.y, fb.width(), fb.height());
    let p2 = Point::from_normalized(v2.x, v2.y, fb.width(), fb.height());

    let (tl, br) = get_triangle_bounding_box(&p0, &p1, &p2);

    for y in tl.y.clamp(0, fb.width() as i32 - 1)..=br.y.clamp(0, fb.width() as i32 - 1) {
        for x in tl.x.clamp(0, fb.width() as i32 - 1)..=br.x.clamp(0, fb.width() as i32 - 1) {
            let (target_x, target_y) = {
                let (nx, ny) = fb.screen_to_normalized(x as u32, y as u32);
                (nx - v0.x, ny - v0.y)
            };
            let newbase_x = inv_col1.x * target_x + inv_col2.x * target_y;
            let newbase_y = inv_col1.y * target_x + inv_col2.y * target_y;
            let local_coords = Vec2::new(newbase_x, newbase_y);

            if local_coords.x >= 0.0
                && local_coords.y >= 0.0
                && (local_coords.x + local_coords.y) <= 1.0
            {
                let ratio_b = newbase_x;
                let ratio_c = newbase_y;
                let ratio_a = 1.0 - ratio_b - ratio_c;
                let fragment_depth = ratio_a * v0.z + ratio_b * v1.z + ratio_c * v2.z;
                if depth_buffer.get_depth(x as u32, y as u32) > fragment_depth {
                    depth_buffer.set_depth(x as u32, y as u32, fragment_depth);
                    fb.set_pixel(x as u32, y as u32, color);
                }
            }
        }
    }
}

fn draw_line_naive<F: Framebuffer>(fb: &mut F, p0: &Point, p1: &Point, color: &Vec3) {
    let diff_x = p1.x - p0.x;
    let diff_y = p1.y - p0.y;
    let stepx = diff_x.signum();
    let stepy = diff_y.signum();

    let additive = diff_y.abs();
    let subtractive = diff_x.max(1);
    let mut accum = additive;
    let mut current_x = p0.x;
    let mut current_y = p0.y;

    //let x 50
    //let y 200
    //let y per x : 0.25
    let mut iter = 0;

    loop {
        fb.set_pixel(current_x as u32, current_y as u32, color);
        if current_x == p1.x && current_y == p1.y {
            break;
        }
        if accum > subtractive {
            current_y += stepy;
            accum -= subtractive;
        } else {
            current_x += stepx;
            accum += additive;
        }
        iter += 1;
    }
    println!("draw_line_naive : {}", iter);
}
