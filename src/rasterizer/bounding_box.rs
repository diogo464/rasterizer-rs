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
        let (minl, maxl) = if self.x <= other.x {
            (self.x, other.x)
        } else {
            (other.x, self.x)
        };
        let (minr, maxr) = if (self.x + self.w) <= (other.x + other.w) {
            (self.x + self.w, other.x + other.w)
        } else {
            (other.x + other.w, self.x + self.w)
        };

        if minr < maxl {
            return None;
        }

        let (mint, maxt) = if self.y <= other.y {
            (self.y, other.y)
        } else {
            (other.y, self.y)
        };
        let (minb, maxb) = if (self.y + self.h) <= (other.y + other.h) {
            (self.y + self.h, other.y + other.h)
        } else {
            (other.y + other.h, self.y + self.h)
        };

        if maxt > minb {
            return None;
        }

        let width = minr - maxl;
        let height = minb - maxt;
        Some(BoundingBox::new(maxl, maxt, width, height))
    }
}
