#![feature(test)]

pub mod model;
pub mod obj;
pub mod ppm;
pub mod shaders;
pub mod texture;

use model::Model;
use rasterizer::math_prelude::*;
use rasterizer::Rasterizer;
use texture::Texture;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::{pixels::PixelFormatEnum, render::TextureAccess};

use shaders::*;

pub struct Camera {
    pub position: Vec3,
    pub fov: f32,
    pub velocity: f32,
    pub target_dir: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub sensitivity: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.9, 0.6, 0.0),
            fov: 90.0f32.to_radians(),
            velocity: 1.0,
            target_dir: Vec3::default(),
            pitch: 0.0,
            yaw: 180.0f32.to_radians(),
            sensitivity: 0.01,
        }
    }
}

impl Camera {
    // 85 deg
    pub const MIN_PITCH: f32 = -1.48352986;
    pub const MAX_PITCH: f32 = 1.48352986;

    // 5 deg
    pub const FOV_MIN: f32 = 0.0872664626;
    // 160 deg
    pub const FOV_MAX: f32 = 2.7925268;

    pub fn generate_matrix(&self) -> Mat4 {
        self.generate_projection_matrix() * self.generate_view_matrix()
    }

    pub fn generate_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, 16.0 / 9.0, 0.1, 100.0)
    }

    pub fn generate_view_matrix(&self) -> Mat4 {
        let fwd = self.calculate_forward();
        Mat4::look_at_rh(self.position, self.position + fwd, Vec3::Y)
    }

    pub fn handle_event(&mut self, event: &Event) {
        match event {
            Event::MouseMotion { xrel, yrel, .. } => {
                self.yaw += *xrel as f32 * self.sensitivity;
                self.pitch -= *yrel as f32 * self.sensitivity;
                self.pitch = self.pitch.clamp(Self::MIN_PITCH, Self::MAX_PITCH);
            }
            Event::KeyDown {
                keycode,
                repeat: false,
                ..
            } => {
                let kc = keycode.unwrap();
                match kc {
                    Keycode::W => self.target_dir.z += 1.0,
                    Keycode::S => self.target_dir.z -= 1.0,
                    Keycode::A => self.target_dir.x -= 1.0,
                    Keycode::D => self.target_dir.x += 1.0,
                    Keycode::Q => self.velocity -= 1.0,
                    Keycode::E => self.velocity += 1.0,
                    Keycode::Space => self.target_dir.y += 1.0,
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
                    Keycode::W => self.target_dir.z -= 1.0,
                    Keycode::S => self.target_dir.z += 1.0,
                    Keycode::A => self.target_dir.x += 1.0,
                    Keycode::D => self.target_dir.x -= 1.0,
                    Keycode::Space => self.target_dir.y -= 1.0,
                    _ => {}
                };
            }
            Event::MouseWheel { y, .. } => {
                self.fov += *y as f32 * 4.0;
                self.fov = self.fov.clamp(Self::FOV_MIN, Self::FOV_MAX);
            }
            _ => {}
        }
    }

    pub fn update(&mut self, dt: f32) {
        let fwd = self.calculate_forward();
        let right = fwd.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(fwd).normalize();
        self.position +=
            (self.target_dir.z * fwd + self.target_dir.x * right + self.target_dir.y * up)
                * dt
                * self.velocity;
    }

    fn calculate_forward(&self) -> Vec3 {
        let x = self.yaw.cos() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = self.yaw.sin() * self.pitch.cos();
        Vec3::new(x, y, z)
    }
}

pub struct ModelViewer {
    pub camera: Camera,
    pub rasterizer: Rasterizer,
}

pub trait Scene {
    #[allow(unused)]
    fn handle_event(&mut self, event: &Event) {}
    fn render(&mut self, model_viewer: &mut ModelViewer);
}

pub struct PBRScene {
    model: Model,
    uniform: PBRUniform,
    vertex: PBRVertexShader,
    fragment: PBRFragmentShader,
}

impl PBRScene {
    pub fn new(
        model: &str,
        albedo: &str,
        normal: &str,
        ao: &str,
        rough: &str,
        metal: &str,
    ) -> Self {
        let model = obj::read_model(model);
        let uniform = PBRUniform {
            model: Default::default(),
            view: Default::default(),
            projection: Default::default(),
            viewpos: Default::default(),
            light_pos: Default::default(),
            albedo_tex: Texture::load(albedo),
            normal_tex: Texture::load(normal),
            ao_tex: Texture::load(ao),
            roughness_tex: Texture::load(rough),
            metallic_tex: Texture::load(metal),
        };

        Self {
            model,
            uniform,
            vertex: PBRVertexShader,
            fragment: PBRFragmentShader,
        }
    }

    pub fn create_cerberus() -> Self {
        Self::new(
            "models/cerberus/cerberus_mesh.obj",
            "models/cerberus/cerberus_albedo.png",
            "models/cerberus/cerberus_normal.png",
            "models/cerberus/cerberus_ao.png",
            "models/cerberus/cerberus_rough.png",
            "models/cerberus/cerberus_metal.png",
        )
    }

    pub fn create_firehydrant() -> Self {
        Self::new(
            "models/firehydrant/firehydrant_mesh.obj",
            "models/firehydrant/firehydrant_albedo.png",
            "models/firehydrant/firehydrant_normal.png",
            "models/firehydrant/firehydrant_ao.png",
            "models/firehydrant/firehydrant_rough.png",
            "models/firehydrant/firehydrant_metal.png",
        )
    }
}

impl Scene for PBRScene {
    fn render(&mut self, model_viewer: &mut ModelViewer) {
        self.uniform.view = model_viewer.camera.generate_view_matrix();
        self.uniform.model = Mat4::IDENTITY;
        self.uniform.projection = model_viewer.camera.generate_projection_matrix();
        self.uniform.viewpos = model_viewer.camera.position;
        self.uniform.light_pos = Vec3::new(0.3, 1.2, 0.4);

        model_viewer.rasterizer.render_mesh(
            &self.model.vertices,
            &self.model.indices,
            &self.vertex,
            &self.fragment,
            &self.uniform,
        );
    }
}

fn main() {
    const WIDTH: u32 = 1280;
    const HEIGHT: u32 = 720;

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

    let mut model_viewer = ModelViewer {
        camera: Default::default(),
        rasterizer: Rasterizer::new(WIDTH, HEIGHT),
    };
    let mut scene: Box<dyn Scene> = Box::new(PBRScene::create_firehydrant());

    let mut pixels: [u8; (4 * WIDTH * HEIGHT) as usize] = [0; (4 * WIDTH * HEIGHT) as usize];
    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut timer = std::time::Instant::now();
    'running: loop {
        let delta = timer.elapsed().as_secs_f32();
        canvas
            .window_mut()
            .set_title(&format!("FPS : {:.02}", 1.0 / delta))
            .unwrap();
        timer = std::time::Instant::now();

        model_viewer.camera.update(delta);
        for event in event_pump.poll_iter() {
            model_viewer.camera.handle_event(&event);
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,
                Event::MouseButtonDown { .. } => {}
                Event::KeyDown {
                    keycode: Some(kc), ..
                } => match kc {
                    Keycode::Num0 => scene = Box::new(PBRScene::create_firehydrant()),
                    Keycode::Num1 => scene = Box::new(PBRScene::create_cerberus()),
                    _ => {}
                },
                _ => {}
            }
        }

        model_viewer.rasterizer.clear();
        scene.render(&mut model_viewer);

        model_viewer
            .rasterizer
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
        print!("\r{:?}", model_viewer.rasterizer.frametime(),);
    }
}
