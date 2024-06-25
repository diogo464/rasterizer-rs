#![feature(test)]

pub mod model;
pub mod obj;
pub mod ppm;
pub mod scene;
pub mod shader;
pub mod texture;

use clap::{Parser, ValueEnum};
use glam::Vec3;
use rasterizer::Rasterizer;
use scene::{DiabloScene, ModelViewer, PBRScene, Scene};

use sdl2::event::Event;
use sdl2::keyboard::Keycode;
use sdl2::rect::Rect;
use sdl2::{pixels::PixelFormatEnum, render::TextureAccess};

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum ArgsScene {
    #[default]
    Diablo,
    Cerberus,
}

#[derive(Debug, Parser)]
struct Args {
    #[clap(long, default_value = "1280")]
    width: u32,

    #[clap(long, default_value = "720")]
    height: u32,

    #[clap(long, default_value = "diablo")]
    scene: ArgsScene,
}

fn main() {
    let args = Args::parse();
    let mut scene: Box<dyn Scene> = match args.scene {
        ArgsScene::Diablo => Box::new(DiabloScene::load()),
        ArgsScene::Cerberus => Box::new(PBRScene::load()),
    };
    run(args.width, args.height, &mut *scene)
}

fn run(width: u32, height: u32, scene: &mut dyn Scene) {
    let sdl_context = sdl2::init().unwrap();
    let video_subsystem = sdl_context.video().unwrap();
    sdl_context.mouse().set_relative_mouse_mode(true);
    sdl_context.mouse().show_cursor(false);

    let window = video_subsystem
        .window("Model viewer", width, height)
        .position_centered()
        .build()
        .unwrap();

    let mut canvas = window.into_canvas().build().unwrap();
    let fullscreen_rect = Rect::new(0, 0, width, height);
    let texture_creator = canvas.texture_creator();
    let mut display_texture = texture_creator
        .create_texture(
            PixelFormatEnum::RGBA32,
            TextureAccess::Streaming,
            width,
            height,
        )
        .expect("Failed to create texture");

    let mut model_viewer = ModelViewer {
        camera: Default::default(),
        light_position: Vec3::new(0.0, 0.0, 3.0),
        rasterizer: Rasterizer::new(width, height),
    };

    let mut pixels = vec![0u8; (4 * width * height) as usize];
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
            .update(fullscreen_rect, &pixels, (4 * width) as usize)
            .unwrap();
        canvas
            .copy(&display_texture, fullscreen_rect, fullscreen_rect)
            .unwrap();
        canvas.present();
        print!("\r{:?}", model_viewer.rasterizer.frametime(),);
    }
}
