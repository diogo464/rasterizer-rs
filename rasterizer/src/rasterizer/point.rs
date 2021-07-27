use crate::math_prelude::*;

#[derive(Debug)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    #[allow(dead_code)]
    pub fn from_normalized(mut x: f32, mut y: f32, width: u32, height: u32) -> Self {
        x = (x + 1.0) / 2.0;
        y = (-y + 1.0) / 2.0;

        let x = ((x * width as f32) as i32).min(width as i32 - 1);
        let y = ((y * height as f32) as i32).min(height as i32 - 1);

        Point::new(x, y)
    }

    #[allow(dead_code)]
    pub fn as_vec2(&self) -> Vec2 {
        Vec2::new(self.x as f32, self.y as f32)
    }
}

impl Default for Point {
    fn default() -> Self {
        Self::new(0, 0)
    }
}
