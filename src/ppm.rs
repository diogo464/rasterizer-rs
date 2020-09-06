use itertools::Itertools;

#[derive(Debug, Default, Copy, Clone)]
struct Color {
    r: f32,
    g: f32,
    b: f32,
}

pub struct PPMImage {
    width: u32,
    height: u32,
    pixels: Vec<Color>,
}

impl PPMImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            pixels: vec![Color::default(); (width * height) as usize],
        }
    }

    pub fn set_pixel(&mut self, x: u32, y: u32, r: f32, g: f32, b: f32) {
        self.pixels[(x + y * self.width) as usize] = Color { r, g, b };
    }

    pub fn pixels(&self) -> impl Iterator<Item = (u32, u32)> {
        (0..self.width).cartesian_product(0..self.height)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn write_to<W: std::io::Write>(&self, mut w: W) -> std::io::Result<()> {
        use std::fmt::Write;

        let mut contents = String::new();
        write!(contents, "P3\n{} {}\n255\n", self.width, self.height).unwrap();
        for y in 0..self.height {
            for x in 0..self.width {
                let color = self.pixels[(x + y * self.width) as usize];
                write!(
                    contents,
                    "{:.0} {:.0} {:.0} ",
                    color.r * 255.0,
                    color.g * 255.0,
                    color.b * 255.0
                )
                .unwrap();
            }
            write!(contents, "\n").unwrap();
        }

        w.write_all(contents.as_bytes())
    }
}
