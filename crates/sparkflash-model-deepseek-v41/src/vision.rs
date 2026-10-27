//! The V4.1 image front end behind the model contract.
//!
//! The arithmetic stays in sparkflash-vision; this adapter maps the contract's options and
//! prepared images to the V4.1 geometry and types.

use std::sync::Arc;

use sparkflash_model::vision::{
    ImageGrid, ImageSpan, PreparedImage, VisionCodec, VisionError, VisionOptions,
};
use sparkflash_vision::prep::PreparedImage as V41PreparedImage;
use sparkflash_vision::{DecodeLimits, IMAGE_TOKEN_ID, ImageFormat, PrepOptions};

/// The V4.1 vision front end.
#[derive(Debug, Default, Clone, Copy)]
pub struct V41Vision;

/// The process-wide instance used by the tokenizer seam.
static V41_VISION: V41Vision = V41Vision;

/// Returns the shared V4.1 vision codec.
#[must_use]
pub fn codec() -> &'static V41Vision {
    &V41_VISION
}

fn grid(grid: sparkflash_vision::Grid) -> ImageGrid {
    ImageGrid {
        n_llm_h: grid.n_llm_h,
        n_llm_w: grid.n_llm_w,
        best_height: grid.best_height,
        best_width: grid.best_width,
    }
}

fn prepared(image: V41PreparedImage) -> PreparedImage {
    PreparedImage {
        grid: grid(image.grid),
        pixels: image.pixels,
        key: image.key,
    }
}

fn v41_prepared(image: &PreparedImage) -> V41PreparedImage {
    V41PreparedImage {
        grid: sparkflash_vision::Grid {
            n_llm_h: image.grid.n_llm_h,
            n_llm_w: image.grid.n_llm_w,
            best_height: image.grid.best_height,
            best_width: image.grid.best_width,
        },
        pixels: Arc::clone(&image.pixels),
        key: image.key,
        format: ImageFormat::Png,
        source: (0, 0),
    }
}

fn span(span: sparkflash_vision::ImageSpan) -> ImageSpan {
    ImageSpan {
        start: span.start,
        grid: grid(span.grid),
        key: span.key,
    }
}

fn error(error: sparkflash_vision::VisionError) -> VisionError {
    VisionError::new(error.to_string())
}

impl VisionCodec for V41Vision {
    fn image_token_id(&self) -> u32 {
        IMAGE_TOKEN_ID
    }

    fn prepare(&self, bytes: &[u8], options: &VisionOptions) -> Result<PreparedImage, VisionError> {
        let options = PrepOptions {
            limits: DecodeLimits {
                max_bytes: options.max_bytes,
                max_dimension: options.max_dimension,
            },
            max_long_side: options.max_long_side,
            ..PrepOptions::default()
        };
        sparkflash_vision::prepare(bytes, &options)
            .map(prepared)
            .map_err(error)
    }

    fn expand(
        &self,
        prompt: &[u32],
        images: &[PreparedImage],
    ) -> Result<(Vec<u32>, Vec<ImageSpan>), VisionError> {
        let images: Vec<V41PreparedImage> = images.iter().map(v41_prepared).collect();
        let (ids, spans) = sparkflash_vision::span::expand(prompt, &images).map_err(error)?;
        Ok((ids, spans.into_iter().map(span).collect()))
    }
}
