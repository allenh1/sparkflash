//! Prints one layer's device schedule, for comparing the two passes by eye.

use sparkflash_model::TpPlan;
use sparkflash_model_inkling2::Inkling2Config;
use sparkflash_model_inkling2::device_plan::{InklingDevicePlan, Pass};

fn main() {
    let config = Inkling2Config::tiny();
    let tp = TpPlan::new(1, 0).expect("a world of one");
    for pass in [Pass::Decode, Pass::Prefill] {
        let rows = if pass == Pass::Decode { 1 } else { 4 };
        let plan = InklingDevicePlan::build(&config, tp, pass, rows, 64).expect("a schedule");
        for index in [0usize, 1] {
            let layer = &plan.layers()[index];
            println!("== {pass:?} layer {index} (sliding={})", layer.sliding);
            for step in &layer.steps {
                println!("  {:?} {:?}", step.op, step.target);
            }
        }
        println!("  epilogue:");
        for step in plan.epilogue() {
            println!("    {:?} {:?}", step.op, step.target);
        }
    }
}
