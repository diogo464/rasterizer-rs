//use image::{DynamicImage, GenericImageView};
use rasterizer::math_prelude::*;

pub struct Texture {
    colors: Vec<Vec4>,
    width: usize,
    height: usize,
}

impl Texture {
    pub fn load(path: &str) -> Texture {
        let img = image::open(path).unwrap().to_rgba8();
        let colors = img
            .pixels()
            .map(|p| Vec4::new(p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32) / 255.0)
            .collect();
        let width = img.width() as usize;
        let height = img.height() as usize;

        Self {
            colors,
            width,
            height,
        }
    }

    pub fn sample(&self, uv: Vec2) -> Vec4 {
        let x = uv.x * (self.width as f32);
        let y = (1.0 - uv.y) * (self.height as f32);

        let xf = x.floor();
        let yf = y.floor();

        let xl = (xf as usize) % self.width;
        let xr = (x.ceil() as usize) % self.width;
        let yt = (y.ceil() as usize) % self.height;
        let yb = (yf as usize) % self.height;

        self.color_at(xl, yb)

        //self.color_at(xl, yt) * (x - xf) * (y - yf)
        //    + self.color_at(xr, yt) * (1.0 - (x - xf)) * (y - yf)
        //    + self.color_at(xl, yb) * (x - xf) * (1.0 - (y - yf))
        //    + self.color_at(xr, yb) * (1.0 - (x - xf)) * (1.0 - (y - yf))

        // (self.color_at(xl,yt) + self.color_at(xr, yt) + self.color_at(xl, yb) + self.color_at(xr, yb)) / 4.0
    }

    fn color_at(&self, x: usize, y: usize) -> Vec4 {
        self.colors[x + y * self.width]
    }
}

//pub trait Texture {
//    fn color_at(&self, u: f32, v: f32) -> Vec4;
//}
//
//impl Texture for DynamicImage {
//    fn color_at(&self, u: f32, v: f32) -> Vec4 {
//        let x = ((u * self.width() as f32) as u32) % self.width();
//        let y = (((1.0 - v) * self.height() as f32) as u32) % self.height();
//        if let Some(c) = self.as_rgb8() {
//            let p = c.get_pixel(x, y);
//            Vec4::new(
//                p[0] as f32 / 255.0,
//                p[1] as f32 / 255.0,
//                p[2] as f32 / 255.0,
//                1.0,
//            )
//        } else if let Some(c) = self.as_rgba8() {
//            let p = c.get_pixel(x, y);
//            Vec4::new(
//                p[0] as f32 / 255.0,
//                p[1] as f32 / 255.0,
//                p[2] as f32 / 255.0,
//                p[3] as f32 / 255.0,
//            )
//        } else {
//            panic!("Invalid pixel mode");
//        }
//    }
//}
