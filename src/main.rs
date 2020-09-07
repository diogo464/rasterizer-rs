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
    //rasterizer::test();

    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;

    let mut image = PPMImage::new(WIDTH, HEIGHT);
    let model = obj::read_model("diablo3.obj");
    let img = image::open("diablo3_pose_diffuse.tga").unwrap();
    let texture = Box::new(SimpleTexture::new(img));

    let mut uniform = ShaderData::new();
    let model_mat = glm::translation(&Vec3::new(-0.1, -0.5, -0.45));
    //* glm::rotation(3.14159 / 2.0, &Vec3::new(0.0, 1.0, 0.0));
    let projection = glm::perspective(16.0 / 9.0, 3.1415 / 2.0, 0.1, 100.0);
    uniform.set_indexed(0, ShaderField::Matrix4(model_mat));
    uniform.set_indexed(1, ShaderField::Matrix4(projection));
    uniform.set_indexed(2, ShaderField::Texture(texture));

    let shader = ShaderProgram::new(
        |uni, v, out| {
            let model = uni.get_indexed(0).unwrap().matrix4();
            let proj = uni.get_indexed(1).unwrap().matrix4();
            let position = Vec4::new(v.position.x, v.position.y, v.position.z, 1.0);
            out.set_indexed(4, ShaderField::Vector2(v.texture.unwrap()));
            let glpos = proj * model * position;
            glpos
        },
        |uni, input| {
            let tcoords = input.get_indexed(4).unwrap();
            let tcoords = tcoords.vector2();
            let tex = uni.get_indexed(2).unwrap().texture();
            tex.color(tcoords.x, tcoords.y)
            //Vec4::new(1.0, 0.0, 0.0, 1.0)
        },
    );
    let mut rasterizer = rasterizer::Rasterizer::new(WIDTH, HEIGHT);

    let start = std::time::Instant::now();

    rasterizer.render_model(&model, &shader, &uniform);
    //rasterizer.render_model(&model, &shader, &uniform);
    //rasterizer.render_model(&model, &shader, &uniform);

    let delta = start.elapsed().as_millis();
    println!("Delta time : {}ms", delta);

    for (x, y) in image.pixels() {
        let pixel = rasterizer.framebuffer().get_color(x, y).unwrap();
        image.set_pixel(x, y, pixel.x, pixel.y, pixel.z);
    }

    image
        .write_to(std::fs::File::create("ouput.ppm").unwrap())
        .unwrap();
}
