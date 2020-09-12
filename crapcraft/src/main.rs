#![feature(clamp)]
pub mod crapcraft;

use crapcraft::block::Block;
use rand::RngCore;
use rasterizer::Rasterizer;
use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::{pixels::PixelFormatEnum, render::TextureAccess};

use glm::{Mat4, Vec3};
use nalgebra_glm as glm;

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
            camera_speed: 4.0,
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
                    Keycode::LShift => self.camera_speed *= 4.0,
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
                    Keycode::LShift => self.camera_speed /= 4.0,
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
        glm::perspective(800.0 / 600.0, self.fov.to_radians(), 0.00001, 100.0)
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

struct Game {
    chunk: crapcraft::chunk::Chunk,
    mesh: crapcraft::chunk::ChunkMesh,
}

impl Game {
    fn new() -> Self {
        let mut chunk = crapcraft::chunk::Chunk::filled(Block::Dirt);
        // chunk.blocks_mut().iter_mut().for_each(|b| {
        //     let mut rng = rand::thread_rng();
        //     if rng.next_u32() >= u32::MAX / 2 {
        //         *b = Block::Air;
        //     }
        // });
        let mesh = chunk.generate_mesh();
        println!("Vertex count : {}", mesh.vertices.len());
        Self { chunk, mesh }
    }

    fn draw(&self, renderer: &mut Rasterizer, state: &ModelViewerState) {
        let vshader = crapcraft::shaders::ChunkVertexShader::default();
        let fshader = crapcraft::shaders::ChunkFragmentShader::default();
        let model = glm::translation(&Vec3::new(0.0, 0.0, -1.0));
        let uniform = crapcraft::shaders::ProjViewModel::new(
            &state.generate_projection_matrix(),
            &state.generate_view_matrix(),
            &model,
        );
        renderer.render_model(
            &self.mesh.vertices,
            &self.mesh.indices,
            &vshader,
            &fshader,
            &uniform,
        );
    }
}

fn main() {
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    let mut state = ModelViewerState::new();
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

    let mut game = Game::new();

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
        game.draw(&mut rasterizer, &state);
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
