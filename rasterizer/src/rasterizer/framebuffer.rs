use crate::math_prelude::*;

pub struct Framebuffer {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) color: Vec<Vec3>,
}

impl Framebuffer {
    pub fn new(width: u32, height: u32) -> Self {
        let size = (width * height) as usize;
        let color = vec![Vec3::default(); size];
        Self {
            width,
            height,
            color,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn pixel_coords(&self) -> impl Iterator<Item = (u32, u32)> {
        let size = self.width * self.height;
        let width = self.width;
        let height = self.height;
        (0..size).map(move |index| {
            let x = index % width;
            let y = index / height;
            (x, y)
        })
    }

    pub fn color(&self) -> impl Iterator<Item = (u32, u32, Vec3)> + '_ {
        self.color.iter().enumerate().map(move |(i, c)| {
            let (x, y) = self.index_to_coords(i as u32);
            (x, y, *c)
        })
    }

    pub fn get_color(&self, x: u32, y: u32) -> Option<Vec3> {
        self.color.get(self.coords_to_index(x, y) as usize).copied()
    }

    pub fn set_color(&mut self, x: u32, y: u32, color: Vec3) {
        let index = self.coords_to_index(x, y) as usize;
        self.color[index] = color;
    }

    fn coords_to_index(&self, x: u32, y: u32) -> u32 {
        x + y * self.width
    }

    fn index_to_coords(&self, index: u32) -> (u32, u32) {
        let x = index % self.width;
        let y = index / self.width;
        (x, y)
    }
}
