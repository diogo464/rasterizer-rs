#![feature(test)]

pub mod model;
pub mod obj;
pub mod ppm;
pub mod scene;
pub mod shader;
pub mod texture;

use glam::Vec3;
use rasterizer::Rasterizer;
use scene::{DiabloScene, ModelViewer, Scene};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::{pixels::PixelFormatEnum, render::TextureAccess};

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
        light_position: Vec3::new(0.0, 0.0, 3.0),
        rasterizer: Rasterizer::new(WIDTH, HEIGHT),
    };
    let mut scene: Box<dyn Scene> = Box::new(DiabloScene::load());

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
                Event::MouseButtonDown { .. } => {
                    model_viewer.light_position = model_viewer.camera.position;
                }
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
