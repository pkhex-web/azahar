// Copyright 2026 Azahar Emulator Project
// Licensed under GPLv3 or any later version
// Refer to the LICENSE.txt file included.

#version 450 core

layout(location = 1) in vec4 primary_color;
layout(location = 2) in vec2 texcoord0;
layout(location = 3) in vec2 texcoord1;
layout(location = 4) in vec2 texcoord2;
layout(location = 5) in float texcoord0_w;
layout(location = 6) in vec4 normquat;
layout(location = 7) in vec3 view;

layout(location = 0) out vec4 color;

struct LightSrc {
    vec3 specular_0;
    vec3 specular_1;
    vec3 diffuse;
    vec3 ambient;
    vec3 position;
    vec3 spot_direction;
    float dist_atten_bias;
    float dist_atten_scale;
};

layout(set = 0, binding = 2, std140) uniform fs_data {
    int framebuffer_scale;
    int alphatest_ref;
    float depth_scale;
    float depth_offset;
    float shadow_bias_constant;
    float shadow_bias_linear;
    int scissor_x1;
    int scissor_y1;
    int scissor_x2;
    int scissor_y2;
    int fog_lut_offset;
    int proctex_noise_lut_offset;
    int proctex_color_map_offset;
    int proctex_alpha_map_offset;
    int proctex_lut_offset;
    int proctex_diff_lut_offset;
    float proctex_bias;
    int shadow_texture_bias;
    ivec4 lighting_lut_offset[6];
    vec3 fog_color;
    vec2 proctex_noise_f;
    vec2 proctex_noise_a;
    vec2 proctex_noise_p;
    vec3 lighting_global_ambient;
    LightSrc light_src[8];
    vec4 const_color[6];
    vec4 tev_combiner_buffer_color;
    vec3 tex_lod_bias;
    vec4 tex_border_color[3];
    vec4 blend_color;
    uvec4 fs_config[18];
};

layout(set = 0, binding = 3) uniform samplerBuffer texture_buffer_lut_lf;
layout(set = 0, binding = 4) uniform samplerBuffer texture_buffer_lut_rg;
layout(set = 0, binding = 5) uniform samplerBuffer texture_buffer_lut_rgba;

#if defined(TEX0_ALIASED)
layout(set = 1, binding = 0) uniform sampler2D tex0;
layout(set = 1, binding = 0) uniform samplerCube tex0_cube;
#elif defined(TEX0_CUBE)
layout(set = 1, binding = 0) uniform samplerCube tex0;
#elif defined(TEX0_SHADOW)
layout(set = 1, binding = 0) uniform usampler2D tex0;
#else
layout(set = 1, binding = 0) uniform sampler2D tex0;
#endif
layout(set = 1, binding = 1) uniform sampler2D tex1;
layout(set = 1, binding = 2) uniform sampler2D tex2;

const int CFG_FRAMEBUFFER = 0;
const int CFG_TEXTURE = 8;
const int CFG_BORDER = 9;
const int CFG_TEV = 12;
const int CFG_LIGHTING = 42;
const int CFG_LUT_D0 = 43;
const int CFG_LUT_D1 = 45;
const int CFG_LUT_SP = 47;
const int CFG_LUT_FR = 49;
const int CFG_LUT_RR = 51;
const int CFG_LUT_RG = 53;
const int CFG_LUT_RB = 55;
const int CFG_LIGHTS = 57;
const int CFG_PROCTEX = 61;
const int CFG_PROCTEX_LUT_WIDTH = 62;
const int CFG_PROCTEX_LUT_OFFSET = 63;
const int CFG_PROCTEX_LOD = 67;
const int CFG_TEXTURES_USED = 68;

uint cfg(int word) {
    return fs_config[word >> 2][word & 3];
}

uint field(int word, int offset, int count) {
    return bitfieldExtract(cfg(word), offset, count);
}

bool flag(int word, int offset) {
    return field(word, offset, 1) != 0u;
}

float byteround(float x) {
    return round(x * 255.0) * (1.0 / 255.0);
}

vec3 byteround(vec3 x) {
    return round(x * 255.0) * (1.0 / 255.0);
}

vec4 byteround(vec4 x) {
    return round(x * 255.0) * (1.0 / 255.0);
}

vec3 quaternion_rotate(vec4 q, vec3 v) {
    vec3 val1 = cross(q.xyz, v) + v * q.w;
    return v + cross(q.xyz, val1) * 2.0;
}

bool BorderHit(int unit, vec2 uv) {
    bool enable_s = flag(CFG_BORDER + unit, 0);
    bool enable_t = flag(CFG_BORDER + unit, 1);
    if (enable_s && enable_t) {
        return any(bvec4(uv.x < 0.0, uv.x > 1.0, uv.y < 0.0, uv.y > 1.0));
    }
    if (enable_s) {
        return any(bvec2(uv.x < 0.0, uv.x > 1.0));
    }
    if (enable_t) {
        return any(bvec2(uv.y < 0.0, uv.y > 1.0));
    }
    return false;
}

vec4 SampleLod(sampler2D tex, vec2 uv, int unit) {
    vec2 coord = uv * vec2(textureSize(tex, 0));
    vec2 d = max(abs(dFdx(coord)), abs(dFdy(coord)));
    float lod = log2(max(d.x, d.y)) + tex_lod_bias[unit];
    return textureLod(tex, uv, lod);
}

#if defined(TEX0_SHADOW)
float CompareShadow(uint pixel, uint z) {
    uint d24 = pixel >> 8;
    float s8 = float(pixel & 255u) * (1.0 / 255.0);
    return d24 <= z ? 0.0 : s8;
}

float SampleShadow2D(ivec2 uv, ivec2 size, uint z) {
    if (any(bvec4(lessThan(uv, ivec2(0)), greaterThanEqual(uv, size)))) {
        return 1.0;
    }
    return CompareShadow(texelFetch(tex0, uv, 0).x, z);
}

vec4 SampleShadow() {
    vec2 uv = texcoord0;
    if (!flag(CFG_TEXTURE, 16)) {
        uv = uv / vec2(texcoord0_w, texcoord0_w);
    }
    uint z = uint(max(0, int(min(abs(texcoord0_w), 1.0) * 16777215.0) - shadow_texture_bias));
    ivec2 size = textureSize(tex0, 0);
    vec2 coord = fma(vec2(size), uv, vec2(-0.5));
    vec2 coord_floor = floor(coord);
    vec2 f = coord - coord_floor;
    ivec2 i = ivec2(coord_floor);
    vec2 s_xy = vec2(SampleShadow2D(i, size, z), SampleShadow2D(i + ivec2(1, 0), size, z));
    vec2 s_zw = vec2(SampleShadow2D(i + ivec2(0, 1), size, z), SampleShadow2D(i + ivec2(1, 1), size, z));
    vec2 t = mix(s_xy, s_zw, f.yy);
    return vec4(mix(t.x, t.y, f.x));
}
#endif

vec4 SampleTex0() {
    uint type = field(CFG_TEXTURE, 0, 3);
    if (type == 5u) {
        return vec4(0.0);
    }
    if (BorderHit(0, texcoord0)) {
        return tex_border_color[0];
    }
#if defined(TEX0_ALIASED)
    if (type == 0u) {
        return SampleLod(tex0, texcoord0, 0);
    }
    if (type == 3u) {
        return textureProj(tex0, vec3(texcoord0, texcoord0_w));
    }
    if (type == 1u) {
        return texture(tex0_cube, vec3(texcoord0, texcoord0_w));
    }
#elif defined(TEX0_CUBE)
    if (type == 1u) {
        return texture(tex0, vec3(texcoord0, texcoord0_w));
    }
#elif defined(TEX0_SHADOW)
    if (type == 2u) {
        return SampleShadow();
    }
#else
    if (type == 0u) {
        return SampleLod(tex0, texcoord0, 0);
    }
    if (type == 3u) {
        return textureProj(tex0, vec3(texcoord0, texcoord0_w));
    }
#endif
    return vec4(0.0);
}

vec4 SampleTex1() {
    if (BorderHit(1, texcoord1)) {
        return tex_border_color[1];
    }
    return SampleLod(tex1, texcoord1, 1);
}

vec4 SampleTex2() {
    vec2 uv = flag(CFG_TEXTURE, 3) ? texcoord1 : texcoord2;
    if (BorderHit(2, uv)) {
        return tex_border_color[2];
    }
    return SampleLod(tex2, uv, 2);
}

float ProcTexLookupLUT(int offset, float coord) {
    coord *= 128.0;
    float index_i = clamp(floor(coord), 0.0, 127.0);
    float index_f = coord - index_i;
    vec2 entry = texelFetch(texture_buffer_lut_rg, int(index_i) + offset).rg;
    return clamp(fma(entry.g, index_f, entry.r), 0.0, 1.0);
}

int ProcTexNoiseRand1D(int v) {
    const int table[] = int[](0, 4, 10, 8, 4, 9, 7, 12, 5, 15, 13, 14, 11, 15, 2, 11);
    return ((v % 9 + 2) * 3 & 0xF) ^ table[(v / 9) & 0xF];
}

float ProcTexNoiseRand2D(vec2 point) {
    const int table[] = int[](10, 2, 15, 8, 0, 7, 4, 5, 5, 13, 2, 6, 13, 9, 3, 14);
    int u2 = ProcTexNoiseRand1D(int(point.x));
    int v2 = ProcTexNoiseRand1D(int(point.y));
    v2 += ((u2 & 3) == 1) ? 4 : 0;
    v2 ^= (u2 & 1) * 6;
    v2 += u2 + 10;
    v2 &= 0xF;
    v2 ^= table[u2];
    return fma(float(v2), 2.0 / 15.0, -1.0);
}

float ProcTexNoiseCoef(vec2 x) {
    vec2 grid = (proctex_noise_f * 9.0) * abs(x + proctex_noise_p);
    vec2 point = floor(grid);
    vec2 frac = grid - point;
    float frac_x_y = frac.x + frac.y;
    float g0 = ProcTexNoiseRand2D(point) * frac_x_y;
    float g1 = ProcTexNoiseRand2D(point + vec2(1.0, 0.0)) * (frac_x_y - 1.0);
    float g2 = ProcTexNoiseRand2D(point + vec2(0.0, 1.0)) * (frac_x_y - 1.0);
    float g3 = ProcTexNoiseRand2D(point + vec2(1.0, 1.0)) * (frac_x_y - 2.0);
    float x_noise = ProcTexLookupLUT(proctex_noise_lut_offset, frac.x);
    float y_noise = ProcTexLookupLUT(proctex_noise_lut_offset, frac.y);
    float x0 = mix(g0, g1, x_noise);
    float x1 = mix(g2, g3, x_noise);
    return mix(x0, x1, y_noise);
}

float ProcTexShiftOffset(float v, uint mode, uint clamp_mode) {
    float offset = clamp_mode == 3u ? 1.0 : 0.5;
    int v_i32 = int(v);
    if (mode == 1u) {
        return offset * float((v_i32 / 2) % 2);
    }
    if (mode == 2u) {
        return offset * float(((v_i32 + 1) / 2) % 2);
    }
    return 0.0;
}

float ProcTexClamp(float v, uint mode) {
    switch (mode) {
    case 0u:
        return v > 1.0 ? 0.0 : v;
    case 2u:
        return fract(v);
    case 3u:
        return (int(v) % 2) == 0 ? fract(v) : 1.0 - fract(v);
    case 4u:
        return v > 0.5 ? 1.0 : 0.0;
    default:
        return min(v, 1.0);
    }
}

float ProcTexCombineAndMap(uint combiner, float u, float v, int offset) {
    float u2v2 = fma(u, u, v * v);
    float combined;
    switch (combiner) {
    case 0u:
        combined = u;
        break;
    case 1u:
        combined = u * u;
        break;
    case 2u:
        combined = v;
        break;
    case 3u:
        combined = v * v;
        break;
    case 4u:
        combined = (u + v) * 0.5;
        break;
    case 5u:
        combined = u2v2 * 0.5;
        break;
    case 6u:
        combined = min(sqrt(u2v2), 1.0);
        break;
    case 7u:
        combined = min(u, v);
        break;
    case 8u:
        combined = max(u, v);
        break;
    case 9u:
        combined = min(fma(u + v, 0.5, sqrt(u2v2)) * 0.5, 1.0);
        break;
    default:
        combined = 0.0;
        break;
    }
    return ProcTexLookupLUT(offset, combined);
}

vec4 SampleProcTexColor(float lut_coord, int level, uint filter_mode) {
    int lut_width = int(cfg(CFG_PROCTEX_LUT_WIDTH)) >> level;
    int lut_offset;
    if (level < 4) {
        lut_offset = int(cfg(CFG_PROCTEX_LUT_OFFSET + level));
    } else {
        const int high_offsets[] = int[](0xF0, 0xF8, 0xFC, 0xFE);
        lut_offset = high_offsets[(level - 4) & 3];
    }
    lut_coord *= float(lut_width - 1);
    if (filter_mode == 1u || filter_mode == 3u || filter_mode == 5u) {
        int lut_index_i = int(lut_coord) + lut_offset;
        float lut_index_f = fract(lut_coord);
        return texelFetch(texture_buffer_lut_rgba, lut_index_i + proctex_lut_offset) +
               texelFetch(texture_buffer_lut_rgba, lut_index_i + proctex_diff_lut_offset) *
                   lut_index_f;
    }
    lut_coord += float(lut_offset);
    return texelFetch(texture_buffer_lut_rgba, int(round(lut_coord)) + proctex_lut_offset);
}

vec4 SampleTex3() {
    uint proctex = cfg(CFG_PROCTEX);
    if ((proctex & 1u) == 0u) {
        return vec4(0.0);
    }
    uint coord = bitfieldExtract(proctex, 1, 2);
    uint u_clamp = bitfieldExtract(proctex, 3, 3);
    uint v_clamp = bitfieldExtract(proctex, 6, 3);
    uint color_combiner = bitfieldExtract(proctex, 9, 4);
    uint alpha_combiner = bitfieldExtract(proctex, 13, 4);
    uint filter_mode = bitfieldExtract(proctex, 17, 3);
    bool separate_alpha = bitfieldExtract(proctex, 20, 1) != 0u;
    bool noise = bitfieldExtract(proctex, 21, 1) != 0u;
    uint u_shift_mode = bitfieldExtract(proctex, 22, 2);
    uint v_shift_mode = bitfieldExtract(proctex, 24, 2);
    float lod_min = float(field(CFG_PROCTEX_LOD, 0, 16));
    float lod_max = float(field(CFG_PROCTEX_LOD, 16, 16));

    vec2 uv = abs(coord == 1u ? texcoord1 : (coord == 2u ? texcoord2 : texcoord0));
    vec2 duv = max(abs(dFdx(uv)), abs(dFdy(uv)));
    float bias = float(int(cfg(CFG_PROCTEX_LUT_WIDTH))) * proctex_bias;
    float lod = log2(abs(bias) * (duv.x + duv.y));
    lod = proctex_bias == 0.0 ? 0.0 : lod;
    lod = clamp(lod, max(0.0, lod_min), min(7.0, lod_max));

    float u_shift = ProcTexShiftOffset(uv.y, u_shift_mode, u_clamp);
    float v_shift = ProcTexShiftOffset(uv.x, v_shift_mode, v_clamp);
    if (noise) {
        uv = uv + proctex_noise_a * ProcTexNoiseCoef(uv);
        uv = abs(uv);
    }
    float u = ProcTexClamp(uv.x + u_shift, u_clamp);
    float v = ProcTexClamp(uv.y + v_shift, v_clamp);
    float lut_coord = ProcTexCombineAndMap(color_combiner, u, v, proctex_color_map_offset);

    vec4 final_color;
    if (filter_mode == 0u || filter_mode == 1u) {
        final_color = SampleProcTexColor(lut_coord, 0, filter_mode);
    } else if (filter_mode == 2u || filter_mode == 3u) {
        final_color = SampleProcTexColor(lut_coord, int(round(lod)), filter_mode);
    } else if (filter_mode == 4u || filter_mode == 5u) {
        int lod_i = int(lod);
        float lod_f = fract(lod);
        final_color = mix(SampleProcTexColor(lut_coord, lod_i, filter_mode),
                          SampleProcTexColor(lut_coord, lod_i + 1, filter_mode), lod_f);
    } else {
        final_color = vec4(0.0);
    }
    if (separate_alpha) {
        final_color.a = ProcTexCombineAndMap(alpha_combiner, u, v, proctex_alpha_map_offset);
    }
    return final_color;
}

vec4 tex_color[4];

vec3 normal;
vec3 tangent;
vec3 light_vector;
vec3 spot_dir;
vec3 half_vector;
uint lighting_config;

float LookupLightingLUT(int lut_index, int index, float delta) {
    vec2 entry = texelFetch(texture_buffer_lut_lf,
                            lighting_lut_offset[lut_index >> 2][lut_index & 3] + index).rg;
    return fma(entry.g, delta, entry.r);
}

float LookupLightingLUTUnsigned(int lut_index, float pos) {
    int index = int(clamp(floor(pos * 256.0), 0.0, 255.0));
    float delta = fma(pos, 256.0, -float(index));
    return LookupLightingLUT(lut_index, index, delta);
}

float LookupLightingLUTSigned(int lut_index, float pos) {
    int index = int(clamp(floor(pos * 128.0), -128.0, 127.0));
    float delta = fma(pos, 128.0, -float(index));
    return LookupLightingLUT(lut_index, index < 0 ? index + 256 : index, delta);
}

uint LightBits(uint slot) {
    return bitfieldExtract(cfg(CFG_LIGHTS + int(slot >> 1)), int(slot & 1u) * 16, 16);
}

bool LutEnabled(int lut_word, uint lut_sampler) {
    if ((cfg(lut_word) & 1u) == 0u) {
        return false;
    }
    switch (lut_sampler) {
    case 0u:
        return lighting_config != 1u;
    case 1u:
        return lighting_config != 0u && lighting_config != 1u && lighting_config != 5u;
    case 3u:
        return lighting_config != 0u && lighting_config != 2u && lighting_config != 4u;
    case 4u:
    case 5u:
        return lighting_config == 4u || lighting_config == 5u || lighting_config == 8u;
    case 6u:
        return lighting_config != 3u;
    case 8u:
        return lighting_config != 2u && lighting_config != 3u;
    }
    return false;
}

float GetLutValue(int sampler_index, uint light_num, int lut_word) {
    uint lut = cfg(lut_word);
    uint lut_input = bitfieldExtract(lut, 2, 3);
    float index;
    switch (lut_input) {
    case 0u:
        index = dot(normal, normalize(half_vector));
        break;
    case 1u:
        index = dot(normalize(view), normalize(half_vector));
        break;
    case 2u:
        index = dot(normal, normalize(view));
        break;
    case 3u:
        index = dot(light_vector, normal);
        break;
    case 4u:
        index = dot(light_vector, spot_dir);
        break;
    case 5u:
        if (lighting_config == 8u) {
            vec3 normalized_half_vector = normalize(half_vector);
            vec3 half_angle_proj =
                normalized_half_vector - normal * dot(normal, normalized_half_vector);
            index = dot(half_angle_proj, tangent);
        } else {
            index = 0.0;
        }
        break;
    default:
        index = 0.0;
        break;
    }
    float value;
    if ((lut & 2u) != 0u) {
        index = bitfieldExtract(LightBits(light_num), 4, 1) != 0u ? abs(index) : max(index, 0.0);
        value = LookupLightingLUTUnsigned(sampler_index, index);
    } else {
        value = LookupLightingLUTSigned(sampler_index, index);
    }
    return uintBitsToFloat(cfg(lut_word + 1)) * value;
}

void WriteLighting(out vec4 primary_fragment_color, out vec4 secondary_fragment_color) {
    uint lighting = cfg(CFG_LIGHTING);
    uint src_num = bitfieldExtract(lighting, 1, 4);
    uint bump_mode = bitfieldExtract(lighting, 5, 2);
    uint bump_selector = bitfieldExtract(lighting, 7, 2);
    bool bump_renorm = bitfieldExtract(lighting, 9, 1) != 0u;
    bool clamp_highlights_enable = bitfieldExtract(lighting, 10, 1) != 0u;
    lighting_config = bitfieldExtract(lighting, 11, 4);
    bool primary_alpha = bitfieldExtract(lighting, 15, 1) != 0u;
    bool secondary_alpha = bitfieldExtract(lighting, 16, 1) != 0u;
    bool enable_shadow = bitfieldExtract(lighting, 17, 1) != 0u;
    bool shadow_primary = bitfieldExtract(lighting, 18, 1) != 0u;
    bool shadow_secondary = bitfieldExtract(lighting, 19, 1) != 0u;
    bool shadow_invert = bitfieldExtract(lighting, 20, 1) != 0u;
    bool shadow_alpha = bitfieldExtract(lighting, 21, 1) != 0u;
    uint shadow_selector = bitfieldExtract(lighting, 22, 2);

    vec4 diffuse_sum = vec4(0.0, 0.0, 0.0, 1.0);
    vec4 specular_sum = vec4(0.0, 0.0, 0.0, 1.0);
    float clamp_highlights = 1.0;
    float geo_factor = 1.0;

    vec3 surface_normal = vec3(0.0, 0.0, 1.0);
    vec3 surface_tangent = vec3(1.0, 0.0, 0.0);
    if (bump_mode == 1u) {
        surface_normal = tex_color[bump_selector].rgb * 2.0 - vec3(1.0);
        if (bump_renorm) {
            float val = 1.0 - fma(surface_normal.x, surface_normal.x,
                                  surface_normal.y * surface_normal.y);
            surface_normal.z = sqrt(max(val, 0.0));
        }
    } else if (bump_mode == 2u) {
        surface_tangent = tex_color[bump_selector].rgb * 2.0 - vec3(1.0);
    }

    vec4 normalized_normquat = normalize(normquat);
    normal = quaternion_rotate(normalized_normquat, surface_normal);
    tangent = quaternion_rotate(normalized_normquat, surface_tangent);

    vec4 shadow = vec4(1.0);
    if (enable_shadow) {
        shadow = tex_color[shadow_selector];
        if (shadow_invert) {
            shadow = vec4(1.0) - shadow;
        }
    }

    for (uint slot = 0u; slot < src_num; slot++) {
        uint light = LightBits(slot);
        uint num = light & 7u;
        bool directional = bitfieldExtract(light, 3, 1) != 0u;
        bool two_sided = bitfieldExtract(light, 4, 1) != 0u;
        bool dist_atten_enable = bitfieldExtract(light, 5, 1) != 0u;
        bool spot_atten_enable = bitfieldExtract(light, 6, 1) != 0u;
        bool geometric_factor_0 = bitfieldExtract(light, 7, 1) != 0u;
        bool geometric_factor_1 = bitfieldExtract(light, 8, 1) != 0u;
        bool light_shadow = bitfieldExtract(light, 9, 1) != 0u;

        vec3 light_position = light_src[num].position;
        light_vector = directional ? light_position : light_position + view;
        float light_distance = length(light_vector);
        light_vector = normalize(light_vector);
        spot_dir = light_src[num].spot_direction;
        half_vector = normalize(view) + light_vector;

        float dot_product = two_sided ? abs(dot(light_vector, normal))
                                      : max(dot(light_vector, normal), 0.0);
        if (clamp_highlights_enable) {
            clamp_highlights = sign(dot_product);
        }

        float spot_atten = 1.0;
        if (spot_atten_enable && LutEnabled(CFG_LUT_SP, 8u)) {
            spot_atten = GetLutValue(8 + int(num), num, CFG_LUT_SP);
        }

        float dist_atten = 1.0;
        if (dist_atten_enable) {
            float index = clamp(fma(light_src[num].dist_atten_scale, light_distance,
                                    light_src[num].dist_atten_bias),
                                0.0, 1.0);
            dist_atten = LookupLightingLUTUnsigned(16 + int(num), index);
        }

        if (geometric_factor_0 || geometric_factor_1) {
            geo_factor = dot(half_vector, half_vector);
            float dot_div_geo = min(dot_product / geo_factor, 1.0);
            geo_factor = geo_factor == 0.0 ? 0.0 : dot_div_geo;
        }

        float d0_lut_value = 1.0;
        if (LutEnabled(CFG_LUT_D0, 0u)) {
            d0_lut_value = GetLutValue(0, num, CFG_LUT_D0);
        }
        vec3 specular_0 = light_src[num].specular_0 * d0_lut_value;
        if (geometric_factor_0) {
            specular_0 = specular_0 * geo_factor;
        }

        float refl_value_r = 1.0;
        if (LutEnabled(CFG_LUT_RR, 6u)) {
            refl_value_r = GetLutValue(6, num, CFG_LUT_RR);
        }
        float refl_value_g = refl_value_r;
        if (LutEnabled(CFG_LUT_RG, 5u)) {
            refl_value_g = GetLutValue(5, num, CFG_LUT_RG);
        }
        float refl_value_b = refl_value_r;
        if (LutEnabled(CFG_LUT_RB, 4u)) {
            refl_value_b = GetLutValue(4, num, CFG_LUT_RB);
        }

        float d1_lut_value = 1.0;
        if (LutEnabled(CFG_LUT_D1, 1u)) {
            d1_lut_value = GetLutValue(1, num, CFG_LUT_D1);
        }
        vec3 refl_value = vec3(refl_value_r, refl_value_g, refl_value_b);
        vec3 specular_1 = (refl_value * d1_lut_value) * light_src[num].specular_1;
        if (geometric_factor_1) {
            specular_1 = specular_1 * geo_factor;
        }

        if (slot == src_num - 1u && LutEnabled(CFG_LUT_FR, 3u)) {
            float value = GetLutValue(3, num, CFG_LUT_FR);
            if (primary_alpha) {
                diffuse_sum.a = value;
            }
            if (secondary_alpha) {
                specular_sum.a = value;
            }
        }

        vec3 diffuse_mul_dot = light_src[num].diffuse * dot_product;
        if (shadow_primary && light_shadow) {
            diffuse_mul_dot = diffuse_mul_dot * shadow.rgb;
        }
        vec3 diffuse_sum_rgb = diffuse_mul_dot + light_src[num].ambient;
        diffuse_sum_rgb = diffuse_sum_rgb * dist_atten;
        diffuse_sum_rgb = diffuse_sum_rgb * spot_atten;

        vec3 specular_sum_rgb = (specular_0 + specular_1) * clamp_highlights;
        specular_sum_rgb = specular_sum_rgb * dist_atten;
        specular_sum_rgb = specular_sum_rgb * spot_atten;
        if (shadow_secondary && light_shadow) {
            specular_sum_rgb = specular_sum_rgb * shadow.rgb;
        }

        diffuse_sum = diffuse_sum + vec4(diffuse_sum_rgb, 0.0);
        specular_sum = specular_sum + vec4(specular_sum_rgb, 0.0);
    }

    if (shadow_alpha) {
        vec4 shadow_a = vec4(1.0, 1.0, 1.0, shadow.a);
        if (primary_alpha) {
            diffuse_sum = diffuse_sum * shadow_a;
        }
        if (secondary_alpha) {
            specular_sum = specular_sum * shadow_a;
        }
    }

    diffuse_sum = diffuse_sum + vec4(lighting_global_ambient, 0.0);
    primary_fragment_color = clamp(diffuse_sum, vec4(0.0), vec4(1.0));
    secondary_fragment_color = clamp(specular_sum, vec4(0.0), vec4(1.0));
}

vec4 rounded_primary_color;
vec4 primary_fragment_color;
vec4 secondary_fragment_color;
vec4 combiner_buffer;
vec4 combiner_output;

vec4 GetSource(uint source, int stage) {
    switch (source) {
    case 0u:
        return rounded_primary_color;
    case 1u:
        return primary_fragment_color;
    case 2u:
        return secondary_fragment_color;
    case 3u:
        return tex_color[0];
    case 4u:
        return tex_color[1];
    case 5u:
        return tex_color[2];
    case 6u:
        return tex_color[3];
    case 13u:
        return combiner_buffer;
    case 14u:
        return const_color[stage];
    case 15u:
        return combiner_output;
    default:
        return vec4(0.0);
    }
}

vec3 ColorModifier(uint modifier, vec4 source) {
    switch (modifier) {
    case 0u:
        return source.rgb;
    case 1u:
        return vec3(1.0) - source.rgb;
    case 2u:
        return source.aaa;
    case 3u:
        return vec3(1.0) - source.aaa;
    case 4u:
        return source.rrr;
    case 5u:
        return vec3(1.0) - source.rrr;
    case 8u:
        return source.ggg;
    case 9u:
        return vec3(1.0) - source.ggg;
    case 12u:
        return source.bbb;
    case 13u:
        return vec3(1.0) - source.bbb;
    default:
        return vec3(1.0);
    }
}

float AlphaModifier(uint modifier, vec4 source) {
    switch (modifier) {
    case 0u:
        return source.a;
    case 1u:
        return 1.0 - source.a;
    case 2u:
        return source.r;
    case 3u:
        return 1.0 - source.r;
    case 4u:
        return source.g;
    case 5u:
        return 1.0 - source.g;
    case 6u:
        return source.b;
    default:
        return 1.0 - source.b;
    }
}

vec3 ColorCombiner(uint op, vec3 r1, vec3 r2, vec3 r3) {
    vec3 result;
    switch (op) {
    case 0u:
        result = r1;
        break;
    case 1u:
        result = r1 * r2;
        break;
    case 2u:
        result = r1 + r2;
        break;
    case 3u:
        result = (r1 + r2) - vec3(0.5);
        break;
    case 4u:
        result = mix(r2, r1, r3);
        break;
    case 5u:
        result = r1 - r2;
        break;
    case 6u:
    case 7u:
        result = vec3(dot(r1 - vec3(0.5), r2 - vec3(0.5)) * 4.0);
        break;
    case 8u:
        result = fma(r1, r2, r3);
        break;
    case 9u:
        result = min(r1 + r2, vec3(1.0)) * r3;
        break;
    default:
        result = vec3(0.0);
        break;
    }
    return clamp(result, vec3(0.0), vec3(1.0));
}

float AlphaCombiner(uint op, float r1, float r2, float r3) {
    float result;
    switch (op) {
    case 0u:
        result = r1;
        break;
    case 1u:
        result = r1 * r2;
        break;
    case 2u:
        result = r1 + r2;
        break;
    case 3u:
        result = (r1 + r2) - 0.5;
        break;
    case 4u:
        result = mix(r2, r1, r3);
        break;
    case 5u:
        result = r1 - r2;
        break;
    case 8u:
        result = fma(r1, r2, r3);
        break;
    case 9u:
        result = min(r1 + r2, 1.0) * r3;
        break;
    default:
        result = 0.0;
        break;
    }
    return clamp(result, 0.0, 1.0);
}

void main() {
    rounded_primary_color = byteround(primary_color);
    primary_fragment_color = vec4(0.0);
    secondary_fragment_color = vec4(0.0);

    uint alpha_test_func = field(CFG_FRAMEBUFFER, 0, 3);
    if (alpha_test_func == 0u) {
        discard;
    }

    float depth = fma(-gl_FragCoord.z, depth_scale, depth_offset);
    if (!flag(CFG_FRAMEBUFFER, 5)) {
        depth = depth / gl_FragCoord.w;
    }
    gl_FragDepth = depth;

    uint scissor_mode = field(CFG_FRAMEBUFFER, 3, 2);
    if (scissor_mode != 0u) {
        bvec2 above = greaterThanEqual(gl_FragCoord.xy, vec2(float(scissor_x1), float(scissor_y1)));
        bvec2 below = lessThan(gl_FragCoord.xy, vec2(float(scissor_x2), float(scissor_y2)));
        bool inside = all(bvec4(above, below));
        if (scissor_mode == 3u ? !inside : inside) {
            discard;
        }
    }

    uint used = cfg(CFG_TEXTURES_USED);
    tex_color[0] = vec4(0.0);
    tex_color[1] = vec4(0.0);
    tex_color[2] = vec4(0.0);
    tex_color[3] = vec4(0.0);
    if ((used & 1u) != 0u) {
        tex_color[0] = SampleTex0();
    }
    if ((used & 2u) != 0u) {
        tex_color[1] = SampleTex1();
    }
    if ((used & 4u) != 0u) {
        tex_color[2] = SampleTex2();
    }
    if ((used & 8u) != 0u) {
        tex_color[3] = SampleTex3();
    }

    if (flag(CFG_LIGHTING, 0)) {
        WriteLighting(primary_fragment_color, secondary_fragment_color);
    }

    combiner_buffer = vec4(0.0);
    vec4 next_combiner_buffer = tev_combiner_buffer_color;
    combiner_output = vec4(0.0);
    uint buffer_input = field(CFG_TEXTURE, 4, 8);

    for (int stage = 0; stage < 6; stage++) {
        int word = CFG_TEV + stage * 4;
        uint sources = cfg(word);
        uint modifiers = cfg(word + 1);
        uint ops = cfg(word + 2);
        uint scales = cfg(word + 3);
        uint color_op = bitfieldExtract(ops, 0, 4);
        uint alpha_op = bitfieldExtract(ops, 16, 4);
        uint color_source1 = bitfieldExtract(sources, 0, 4);
        uint color_source2 = bitfieldExtract(sources, 4, 4);
        uint color_source3 = bitfieldExtract(sources, 8, 4);
        uint alpha_source1 = bitfieldExtract(sources, 16, 4);
        uint alpha_source2 = bitfieldExtract(sources, 20, 4);
        uint alpha_source3 = bitfieldExtract(sources, 24, 4);
        uint color_modifier1 = bitfieldExtract(modifiers, 0, 4);
        uint alpha_modifier1 = bitfieldExtract(modifiers, 12, 3);
        uint color_scale = bitfieldExtract(scales, 0, 2);
        uint alpha_scale = bitfieldExtract(scales, 16, 2);
        float color_multiplier = color_scale < 3u ? float(1u << color_scale) : 1.0;
        float alpha_multiplier = alpha_scale < 3u ? float(1u << alpha_scale) : 1.0;

        bool passthrough = color_op == 0u && alpha_op == 0u && color_source1 == 15u &&
                           alpha_source1 == 15u && color_modifier1 == 0u && alpha_modifier1 == 0u &&
                           color_multiplier == 1.0 && alpha_multiplier == 1.0;
        if (!passthrough) {
            bool first = stage == 0;
            vec3 color_results_1 = ColorModifier(
                color_modifier1,
                GetSource(first && color_source1 == 15u ? color_source3 : color_source1, stage));
            vec3 color_results_2 = ColorModifier(
                bitfieldExtract(modifiers, 4, 4),
                GetSource(first && color_source2 == 15u ? color_source3 : color_source2, stage));
            vec3 color_results_3 =
                ColorModifier(bitfieldExtract(modifiers, 8, 4), GetSource(color_source3, stage));
            vec3 color_output =
                byteround(ColorCombiner(color_op, color_results_1, color_results_2, color_results_3));

            float alpha_output;
            if (color_op == 7u) {
                alpha_output = color_output.r;
            } else {
                float alpha_results_1 = AlphaModifier(
                    alpha_modifier1,
                    GetSource(first && alpha_source1 == 15u ? alpha_source3 : alpha_source1, stage));
                float alpha_results_2 = AlphaModifier(
                    bitfieldExtract(modifiers, 16, 3),
                    GetSource(first && alpha_source2 == 15u ? alpha_source3 : alpha_source2, stage));
                float alpha_results_3 = AlphaModifier(bitfieldExtract(modifiers, 20, 3),
                                                      GetSource(alpha_source3, stage));
                alpha_output = byteround(
                    AlphaCombiner(alpha_op, alpha_results_1, alpha_results_2, alpha_results_3));
            }

            color_output = clamp(color_output * color_multiplier, vec3(0.0), vec3(1.0));
            alpha_output = clamp(alpha_output * alpha_multiplier, 0.0, 1.0);
            combiner_output = vec4(color_output, alpha_output);
        }

        combiner_buffer = next_combiner_buffer;
        if (stage < 4) {
            if (bitfieldExtract(buffer_input, stage, 1) != 0u) {
                next_combiner_buffer.rgb = combiner_output.rgb;
            }
            if (bitfieldExtract(buffer_input, stage + 4, 1) != 0u) {
                next_combiner_buffer.a = combiner_output.a;
            }
        }
    }

    if (alpha_test_func != 1u) {
        int alpha = int(combiner_output.a * 255.0);
        bool kill;
        switch (alpha_test_func) {
        case 2u:
            kill = alpha != alphatest_ref;
            break;
        case 3u:
            kill = alpha == alphatest_ref;
            break;
        case 4u:
            kill = alpha >= alphatest_ref;
            break;
        case 5u:
            kill = alpha > alphatest_ref;
            break;
        case 6u:
            kill = alpha <= alphatest_ref;
            break;
        default:
            kill = alpha < alphatest_ref;
            break;
        }
        if (kill) {
            discard;
        }
    }

    if (field(CFG_TEXTURE, 12, 3) == 5u) {
        float fog_index = (flag(CFG_TEXTURE, 15) ? 1.0 - depth : depth) * 128.0;
        float fog_i = clamp(floor(fog_index), 0.0, 127.0);
        float fog_f = fog_index - fog_i;
        vec2 fog_lut_entry = texelFetch(texture_buffer_lut_lf, int(fog_i) + fog_lut_offset).rg;
        float fog_factor = clamp(fma(fog_f, fog_lut_entry.g, fog_lut_entry.r), 0.0, 1.0);
        combiner_output.rgb = mix(fog_color, combiner_output.rgb, vec3(fog_factor));
    }

    vec4 final_color = byteround(combiner_output);
    uint logic_op = field(CFG_FRAMEBUFFER, 6, 4);
    if (logic_op == 0u) {
        final_color = vec4(0.0);
    } else if (logic_op == 4u) {
        final_color = vec4(1.0);
    }
    color = final_color;
}
