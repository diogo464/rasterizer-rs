use glam::{Mat4, Vec3};
use sdl2::{event::Event, keyboard::Keycode};

pub struct Camera {
    pub position: Vec3,
    pub fov: f32,
    pub velocity: f32,
    pub target_dir: Vec3,
    pub pitch: f32,
    pub yaw: f32,
    pub sensitivity: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            position: Vec3::new(0.9, 0.6, 0.0),
            fov: 90.0f32.to_radians(),
            velocity: 3.0,
            target_dir: Vec3::default(),
            pitch: 0.0,
            yaw: 180.0f32.to_radians(),
            sensitivity: 0.01,
        }
    }
}

impl Camera {
    // 85 deg
    pub const MIN_PITCH: f32 = -1.483_529_8;
    pub const MAX_PITCH: f32 = 1.483_529_8;

    // 5 deg
    pub const FOV_MIN: f32 = 0.087_266_46;
    // 160 deg
    pub const FOV_MAX: f32 = 2.7925268;

    pub fn generate_matrix(&self) -> Mat4 {
        self.generate_projection_matrix() * self.generate_view_matrix()
    }

    pub fn generate_projection_matrix(&self) -> Mat4 {
        Mat4::perspective_rh(self.fov, 16.0 / 9.0, 0.1, 100.0)
    }

    pub fn generate_view_matrix(&self) -> Mat4 {
        let fwd = self.calculate_forward();
        Mat4::look_at_rh(self.position, self.position + fwd, Vec3::Y)
    }

    pub fn handle_event(&mut self, event: &Event) {
        match event {
            Event::MouseMotion { xrel, yrel, .. } => {
                self.yaw += *xrel as f32 * self.sensitivity;
                self.pitch -= *yrel as f32 * self.sensitivity;
                self.pitch = self.pitch.clamp(Self::MIN_PITCH, Self::MAX_PITCH);
            }
            Event::KeyDown {
                keycode,
                repeat: false,
                ..
            } => {
                let kc = keycode.unwrap();
                match kc {
                    Keycode::W => self.target_dir.z += 1.0,
                    Keycode::S => self.target_dir.z -= 1.0,
                    Keycode::A => self.target_dir.x -= 1.0,
                    Keycode::D => self.target_dir.x += 1.0,
                    Keycode::Q => self.velocity -= 1.0,
                    Keycode::E => self.velocity += 1.0,
                    Keycode::Space => self.target_dir.y += 1.0,
                    Keycode::V => self.target_dir.y -= 1.0,
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
                    Keycode::W => self.target_dir.z -= 1.0,
                    Keycode::S => self.target_dir.z += 1.0,
                    Keycode::A => self.target_dir.x += 1.0,
                    Keycode::D => self.target_dir.x -= 1.0,
                    Keycode::Space => self.target_dir.y -= 1.0,
                    Keycode::V => self.target_dir.y += 1.0,
                    _ => {}
                };
            }
            Event::MouseWheel { y, .. } => {
                self.fov += *y as f32 * 4.0;
                self.fov = self.fov.clamp(Self::FOV_MIN, Self::FOV_MAX);
            }
            _ => {}
        }
    }

    pub fn update(&mut self, dt: f32) {
        let fwd = self.calculate_forward();
        let right = fwd.cross(Vec3::new(0.0, 1.0, 0.0)).normalize();
        let up = right.cross(fwd).normalize();
        self.position +=
            (self.target_dir.z * fwd + self.target_dir.x * right + self.target_dir.y * up)
                * dt
                * self.velocity;
    }

    fn calculate_forward(&self) -> Vec3 {
        let x = self.yaw.cos() * self.pitch.cos();
        let y = self.pitch.sin();
        let z = self.yaw.sin() * self.pitch.cos();
        Vec3::new(x, y, z)
    }
}
