#[derive(Debug, Copy, Clone, PartialEq)]
pub struct BoundingBox {
    x: u32,
    y: u32,
    w: u32,
    h: u32,
}

impl BoundingBox {
    pub fn new(x: u32, y: u32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    pub fn x(&self) -> u32 {
        self.x
    }
    pub fn y(&self) -> u32 {
        self.y
    }
    pub fn width(&self) -> u32 {
        self.w
    }
    pub fn height(&self) -> u32 {
        self.h
    }
    pub fn overlap(&self, other: &BoundingBox) -> Option<BoundingBox> {
        let maxl = self.x.max(other.x);
        let minr = (self.x + self.w).min(other.x + other.w);

        if minr < maxl {
            return None;
        }

        let maxt = self.y.max(other.y);
        let minb= (self.y + self.h).min(other.y + other.h);
        
        if maxt > minb {
            return None;
        }

        let width = minr - maxl;
        let height = minb - maxt;
        Some(BoundingBox::new(maxl, maxt, width, height))
    }
}
