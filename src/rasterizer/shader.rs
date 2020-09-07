use super::vertex_data::VertexData;
use glm::{Mat3, Mat4, Vec2, Vec3, Vec4};
use nalgebra_glm as glm;
use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    hash::{Hash, Hasher},
};

pub type VertexShader =
    fn(uniform: &ShaderData, vertex: &VertexData, output: &mut ShaderData) -> Vec4;
pub type FragmentShader = fn(uniform: &ShaderData, input: &ShaderDataIterpolator) -> Vec4;

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
    Texture(Box<dyn Texture + Sync + Send>),
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
    pub fn texture(&self) -> &Box<dyn Texture + Sync + Send> {
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
    fields: [(u64, ShaderField); 8],
}

impl ShaderData {
    const INVALID_HASH: u64 = 0;

    pub fn new() -> Self {
        Self {
            fields: [
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
                (Self::INVALID_HASH, ShaderField::Number(0.0)),
            ],
        }
    }

    pub fn set(&mut self, name: &str, field: ShaderField) {
        let key_hash = Self::hash_key(name);
        for f in self.fields.iter_mut() {
            if f.0 == Self::INVALID_HASH {
                f.0 = key_hash;
                f.1 = field;
                return;
            }
        }
        panic!();
    }

    pub fn set_indexed(&mut self, index: usize, field: ShaderField) {
        self.fields[index] = ((index + 1) as u64, field);
    }

    pub fn get(&self, name: &str) -> Option<&ShaderField> {
        let key_hash = Self::hash_key(name);
        drop(name);
        self.fields
            .iter()
            .find(|f| f.0 == key_hash)
            .and_then(|f| Some(&f.1))
    }

    pub fn get_indexed(&self, index: usize) -> Option<&ShaderField> {
        self.fields.get(index).map(|f| &f.1)
    }

    fn hash_key(key: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for ShaderData {
    fn default() -> Self {
        Self::new()
    }
}

// impl Interpolate for ShaderData {
//     fn interpolate(v0: &Self, v1: &Self, v2: &Self, r0: f32, r1: f32, r2: f32) -> Self {
//         let mut new_fields = HashMap::new();
//         for (k, v) in v0.fields.iter() {
//             let interpolated = ShaderField::interpolate(
//                 v,
//                 &v1.fields.get(k).unwrap(),
//                 &v2.get(k).unwrap(),
//                 r0,
//                 r1,
//                 r2,
//             );
//             new_fields.insert(k.clone(), interpolated);
//         }
//         Self { fields: new_fields }
//     }
// }

pub struct ShaderDataIterpolator<'a> {
    pub vertex0_data: &'a ShaderData,
    pub vertex1_data: &'a ShaderData,
    pub vertex2_data: &'a ShaderData,
    pub vertex0_ratio: f32,
    pub vertex1_ratio: f32,
    pub vertex2_ratio: f32,
}

impl<'a> ShaderDataIterpolator<'a> {
    pub fn get(&self, name: &str) -> Option<ShaderField> {
        let field0 = self.vertex0_data.get(name)?;
        let field1 = self.vertex1_data.get(name)?;
        let field2 = self.vertex2_data.get(name)?;
        Some(ShaderField::interpolate(
            &field0,
            &field1,
            &field2,
            self.vertex0_ratio,
            self.vertex1_ratio,
            self.vertex2_ratio,
        ))
    }

    pub fn get_indexed(&self, index: usize) -> Option<ShaderField> {
        let field0 = self.vertex0_data.get_indexed(index)?;
        let field1 = self.vertex1_data.get_indexed(index)?;
        let field2 = self.vertex2_data.get_indexed(index)?;
        Some(ShaderField::interpolate(
            &field0,
            &field1,
            &field2,
            self.vertex0_ratio,
            self.vertex1_ratio,
            self.vertex2_ratio,
        ))
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

    pub(super) fn run_fragment(&self, uniform: &ShaderData, input: &ShaderDataIterpolator) -> Vec4 {
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
