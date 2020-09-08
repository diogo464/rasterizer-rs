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
    normal: Vec3,
}
impl ShaderData for TextureShaderData {}
impl Interpolate for TextureShaderData {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        Self {
            texture_coords: v0.texture_coords * r0
                + v1.texture_coords * r1
                + v2.texture_coords * r2,
            normal: Vec3::interpolate(&v0.normal, &v1.normal, &v2.normal, r0, r1, r2),
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
            normal: vertex.normal.unwrap(),
        };
        (glpos, data)
    }

    fn fragment(&self, data: &Self::Data) -> Vec4 {
        let shade = (0.3 + data.normal.dot(&Vec3::new(0.5, 1.0, 0.5))).min(1.0);
        <DynamicImage as Texture>::color(
            &self.texture,
            data.texture_coords.x,
            data.texture_coords.y,
        ) * shade
    }
}

struct ModelViewerState {
    sensitivity: f32,
    camera_speed: f32,
    fov: f32,
    position: Vec3,
    velocity: Vec3,
    pitch: f32,
    yaw: f32,
}

impl ModelViewerState {
    const MIN_ANGLE: f32 = -85.0;
    const MAX_ANGLE: f32 = 85.0;

    fn new() -> Self {
        Self {
            sensitivity: 0.01,
            camera_speed: 1.0,
            fov: 90.0,
            position: Vec3::new(0.0, 0.0, 2.0),
            velocity: Vec3::new(0.0, 0.0, 0.0),
            pitch: 0.0,
            yaw: 0.0,
        }
    }

    fn handle_event(&mut self, ev: &Event) {
        match ev {
            Event::MouseMotion { xrel, yrel, .. } => {
                self.yaw += *xrel as f32 * self.sensitivity;
                self.pitch -= *yrel as f32 * self.sensitivity;
                self.pitch = self
                    .pitch
                    .clamp(Self::MIN_ANGLE.to_radians(), Self::MAX_ANGLE.to_radians());
            }
            Event::KeyDown {
                keycode,
                repeat: false,
                ..
            } => {
                let kc = keycode.unwrap();
                match kc {
                    Keycode::W => self.velocity.z += 1.0,
                    Keycode::S => self.velocity.z -= 1.0,
                    Keycode::A => self.velocity.x -= 1.0,
                    Keycode::D => self.velocity.x += 1.0,
                    Keycode::Space => self.velocity.y += 1.0,
                    _ => {}
                };
            }
            Event::KeyUp {
                keycode,
                repeat: false,
                ..
            } => {
                let kc = keycode.unwrap();
                match kc {
                    Keycode::W => self.velocity.z -= 1.0,
                    Keycode::S => self.velocity.z += 1.0,
                    Keycode::A => self.velocity.x += 1.0,
                    Keycode::D => self.velocity.x -= 1.0,
                    Keycode::Space => self.velocity.y -= 1.0,
                    _ => {}
                };
            }
            Event::MouseWheel { y, .. } => {
                self.fov += *y as f32 * 10.0;
            }
            _ => {}
        }
    }

    fn update(&mut self, delta: f32) {
        let fwd = self.calculate_forward();
        let right = fwd.cross(&Vec3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(&fwd).normalize();
        self.position += (self.velocity.z * fwd + self.velocity.x * right + self.velocity.y * up)
            * delta
            * self.camera_speed;
    }

    fn generate_projection_matrix(&self) -> Mat4 {
        glm::perspective(16.0 / 9.0, self.fov.to_radians(), 0.001, 100.0)
    }

    fn generate_view_matrix(&self) -> Mat4 {
        let fwd = self.calculate_forward();
        glm::look_at(
            &self.position,
            &(self.position + fwd),
            &Vec3::new(0.0, 1.0, 0.0),
        )
    }

    fn calculate_forward(&self) -> Vec3 {
        let x = self.yaw.cos() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = self.yaw.sin() * self.pitch.cos();
        Vec3::new(x, y, z)
    }
}

fn main() {
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    let model = obj::read_model("diablo3.obj");
    let img = image::open("diablo3_pose_diffuse.tga").unwrap();
    let mut state = ModelViewerState::new();

    let model_mat = glm::translation(&Vec3::new(-0.1, -0.5, -0.45));
    let projection = glm::perspective(16.0 / 9.0, 3.1415 / 2.0, 0.001, 100.0);
    let mut shader = TextureShader {
        projection,
        view: state.generate_view_matrix(),
        model: model_mat,
        texture: img,
    };

    let mut rasterizer = rasterizer::Rasterizer::new(WIDTH, HEIGHT);

    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    sdl_context.mouse().set_relative_mouse_mode(true);
    sdl_context.mouse().show_cursor(false);

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
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut timer = std::time::Instant::now();
    'running: loop {
        let delta = timer.elapsed().as_secs_f32();
        timer = std::time::Instant::now();

        state.update(delta);
        for event in event_pump.poll_iter() {
            state.handle_event(&event);
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                _ => {}
            }
        }

        rasterizer.clear();

        shader.view = state.generate_view_matrix();
        shader.projection = state.generate_projection_matrix();
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
        print!(
            "\r{:?} {:?}",
            rasterizer.frametime(),
            rasterizer.frametime().total()
        );
    }
}
