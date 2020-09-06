use super::vertex_data::VertexData;
use glm::{Mat3, Mat4, Vec2, Vec3, Vec4};
use nalgebra_glm as glm;
use std::collections::HashMap;

pub type VertexShader =
    fn(uniform: &ShaderData, vertex: &VertexData, output: &mut ShaderData) -> Vec4;
pub type FragmentShader = fn(uniform: &ShaderData, input: &ShaderData) -> Vec4;

pub(super) trait Interpolate {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self;
}

pub trait Texture {
    fn color(&self, u: f32, v: f32) -> Vec4;
}

pub enum ShaderField {
    Number(f32),
    Vector2(Vec2),
    Vector3(Vec3),
    Vector4(Vec4),
    Matrix3(Mat3),
    Matrix4(Mat4),
    Texture(Box<dyn Texture>),
}

impl ShaderField {
    pub fn number(&self) -> &f32 {
        if let Self::Number(v) = self {
            return v;
        } else {
            panic!();
        }
    }
    pub fn vector2(&self) -> &Vec2 {
        if let Self::Vector2(v) = self {
            return v;
        } else {
            panic!();
        }
    }
    pub fn vector3(&self) -> &Vec3 {
        if let Self::Vector3(v) = self {
            return v;
        } else {
            panic!();
        }
    }
    pub fn vector4(&self) -> &Vec4 {
        if let Self::Vector4(v) = self {
            return v;
        } else {
            panic!();
        }
    }
    pub fn matrix3(&self) -> &Mat3 {
        if let Self::Matrix3(v) = self {
            return v;
        } else {
            panic!();
        }
    }
    pub fn matrix4(&self) -> &Mat4 {
        if let Self::Matrix4(v) = self {
            return v;
        } else {
            panic!();
        }
    }
    pub fn texture(&self) -> &Box<dyn Texture> {
        if let Self::Texture(v) = self {
            return v;
        } else {
            panic!();
        }
    }
}

impl Interpolate for ShaderField {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        match (v0, v1, v2) {
            (Self::Number(n0), Self::Number(n1), Self::Number(n2)) => {
                Self::Number(n0 * r0 + n1 * r1 + n2 * r2)
            }
            (Self::Vector2(n0), Self::Vector2(n1), Self::Vector2(n2)) => {
                Self::Vector2(n0 * r0 + n1 * r1 + n2 * r2)
            }
            (Self::Vector3(n0), Self::Vector3(n1), Self::Vector3(n2)) => {
                Self::Vector3(n0 * r0 + n1 * r1 + n2 * r2)
            }
            (Self::Vector4(n0), Self::Vector4(n1), Self::Vector4(n2)) => {
                Self::Vector4(n0 * r0 + n1 * r1 + n2 * r2)
            }
            _ => panic!("Interpolation not defined"),
        }
    }
}

pub struct ShaderData {
    fields: HashMap<String, ShaderField>,
}

impl ShaderData {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: &str, field: ShaderField) {
        self.fields.insert(name.to_owned(), field);
    }

    pub fn get(&self, name: &str) -> Option<&ShaderField> {
        self.fields.get(name)
    }
}

impl Default for ShaderData {
    fn default() -> Self {
        Self::new()
    }
}

impl Interpolate for ShaderData {
    fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
        let mut new_fields = HashMap::new();
        for (k, v) in v0.fields.iter() {
            let interpolated = ShaderField::interpolate(
                v,
                &v1.fields.get(k).unwrap(),
                &v2.get(k).unwrap(),
                r0,
                r1,
                r2,
            );
            new_fields.insert(k.clone(), interpolated);
        }
        Self { fields: new_fields }
    }
}

pub struct ShaderProgram {
    vertex: VertexShader,
    fragment: FragmentShader,
}

impl ShaderProgram {
    pub fn new(vertex: VertexShader, fragment: FragmentShader) -> Self {
        Self { vertex, fragment }
    }

    pub fn from_vertex(vertex: VertexShader) -> Self {
        Self {
            vertex,
            fragment: |_, _| Vec4::new(1.0, 0.0, 0.0, 1.0),
        }
    }

    pub(super) fn run_vertex(
        &self,
        uniform: &ShaderData,
        vertex: &VertexData,
    ) -> (Vec3, ShaderData) {
        let mut data = ShaderData::new();
        let pos = (&self.vertex)(uniform, vertex, &mut data);
        (pos.xyz() / pos[3], data)
    }

    pub(super) fn run_fragment(&self, uniform: &ShaderData, input: &ShaderData) -> Vec4 {
        (&self.fragment)(uniform, input)
    }
}

impl Default for ShaderProgram {
    fn default() -> Self {
        Self::new(
            |_, v, _| Vec4::new(v.position.x, v.position.y, v.position.z, 1.0),
            |_, _| Vec4::new(1.0, 0.0, 0.0, 1.0),
        )
    }
}
