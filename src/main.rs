#![feature(test)]
#![feature(clamp)]
#![feature(slice_fill)]

pub mod obj;
pub mod rasterizer;

use crate::rasterizer::Interpolate;
use rasterizer::{Model, Shader, ShaderData, Texture, VertexData};
use sdl2::rect::Rect;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::{
    pixels::{Color, PixelFormatEnum},
    render::TextureAccess,
};
use std::time::Duration;

mod ppm;
use glm::{Mat4, Vec2, Vec3, Vec4};
use image::{DynamicImage, GenericImageView};
use nalgebra_glm as glm;
use ppm::PPMImage;

impl Texture for DynamicImage {
    fn color(&self, u: f32, v: f32) -> Vec4 {
        let x = (u * self.width() as f32) as u32;
        let y = ((1.0 - v) * self.height() as f32) as u32;
        let c = self.as_rgb8().unwrap();
        let p = c.get_pixel(x, y);
        Vec4::new(
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
            1.0,
        )
    }
}

struct TextureShaderData {
    texture_coords: Vec2,
}
impl ShaderData for TextureShaderData {}
impl Interpolate for TextureShaderData {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        Self {
            texture_coords: v0.texture_coords * r0
                + v1.texture_coords * r1
                + v2.texture_coords * r2,
        }
    }
}

struct TextureShader {
    projection: Mat4,
    view: Mat4,
    model: Mat4,
    texture: DynamicImage,
}

impl Shader for TextureShader {
    type Data = TextureShaderData;

    fn vertex(&self, vertex: &VertexData) -> (Vec4, Self::Data) {
        let position = Vec4::new(vertex.position.x, vertex.position.y, vertex.position.z, 1.0);
        let glpos = self.projection * self.view * self.model * position;
        let data = TextureShaderData {
            texture_coords: vertex.texture.unwrap(),
        };
        (glpos, data)
    }

    fn fragment(&self, data: &Self::Data) -> Vec4 {
        <DynamicImage as Texture>::color(
            &self.texture,
            data.texture_coords.x,
            data.texture_coords.y,
        )
    }
}

struct ModelViewerState {
    sensitivity: f32,
    position: Vec3,
    pitch: f32,
    yaw: f32,
}

impl ModelViewerState {
    fn new() -> Self {
        Self {
            sensitivity: 0.01,
            position: Vec3::new(0.0, 0.0, 2.0),
            pitch: 0.0,
            yaw: 0.0,
        }
    }

    fn handle_mouse_move(&mut self, xrel: i32, yrel: i32) {
        self.yaw += xrel as f32 * self.sensitivity;
        self.pitch -= yrel as f32 * self.sensitivity;
    }

    fn generate_view_matrix(&self) -> Mat4 {
        let x = self.yaw.cos() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = self.yaw.sin() * self.pitch.cos();
        glm::look_at(
            &self.position,
            &(self.position + Vec3::new(x, y, z)),
            &Vec3::new(0.0, 1.0, 0.0),
        )
    }
}

fn main() {
    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;

    let model = obj::read_model("diablo3.obj");
    let img = image::open("diablo3_pose_diffuse.tga").unwrap();
    let mut state = ModelViewerState::new();

    let model_mat = glm::translation(&Vec3::new(-0.1, -0.5, -0.45));
    let projection = glm::perspective(16.0 / 9.0, 3.1415 / 2.0, 0.1, 100.0);
    let mut shader = TextureShader {
        projection,
        view: state.generate_view_matrix(),
        model: model_mat,
        texture: img,
    };

    let mut rasterizer = rasterizer::Rasterizer::new(WIDTH, HEIGHT);

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();

    let window = video_subsystem
        .window("Model viewer", WIDTH, HEIGHT)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let fullscreen_rect = Rect::new(0, 0, WIDTH, HEIGHT);
    let texture_creator = canvas.texture_creator();
    let mut display_texture = texture_creator
        .create_texture(
            PixelFormatEnum::RGBA32,
            TextureAccess::Streaming,
            WIDTH,
            HEIGHT,
        )
        .expect("Failed to create texture");

    let mut pixels: [u8; (4 * WIDTH * HEIGHT) as usize] = [0; (4 * WIDTH * HEIGHT) as usize];
    let mut velocity = Vec3::new(0.0, 0.0, 0.0);
    let mut event_pump = sdl_context.event_pump().unwrap();
    'running: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::MouseMotion { xrel, yrel, .. } => {
                    state.handle_mouse_move(xrel, yrel);
                }
                Event::KeyDown { keycode, .. } => {
                    let kc = keycode.unwrap();
                    match kc {
                        Keycode::W => velocity.z = -1.0,
                        Keycode::S => velocity.z = 1.0,
                        Keycode::A => velocity.x = -1.0,
                        Keycode::D => velocity.x = 1.0,
                        _ => {}
                    };
                }
                Event::KeyUp { keycode, .. } => {
                    let kc = keycode.unwrap();
                    match kc {
                        Keycode::W | Keycode::S => velocity.z = 0.0,
                        Keycode::A | Keycode::D => velocity.x = 0.0,
                        _ => {}
                    };
                }
                _ => {}
            }
        }

        rasterizer.clear();

        state.position += velocity * 0.016;
        shader.view = state.generate_view_matrix();
        rasterizer.render_model(&model, &shader);

        rasterizer
            .framebuffer()
            .color()
            .enumerate()
            .for_each(|(index, (_, _, c))| {
                pixels[index * 4] = (c.x * 255.0) as u8;
                pixels[index * 4 + 1] = (c.y * 255.0) as u8;
                pixels[index * 4 + 2] = (c.z * 255.0) as u8;
                pixels[index * 4 + 3] = 255;
            });

        display_texture
            .update(fullscreen_rect, &pixels, (4 * WIDTH) as usize)
            .unwrap();
        canvas
            .copy(&display_texture, fullscreen_rect, fullscreen_rect)
            .unwrap();
        canvas.present();
        print!("\r{:?}", rasterizer.frametime().total());
        ::std::thread::sleep(Duration::new(0, 1_000_000_000u32 / 60));
    }
}
