//! Request-ID targeting for trace capture.
//!
//! This lives in its own integration test binary so the process-global filter cannot leak into
//! the unit tests. That is not enough on its own: tests inside ONE binary still run on parallel
//! threads by default, so every test here takes TEST_LOCK before touching the filter - without
//! it, one test's set_request_filter races another test's "no filter installed" assertion and
//! the workspace gate goes nondeterministically red.

use sparkflash_trace::{
    TraceScope, active_mask, arm, bind_current_request, disarm, is_request_traced, request_filter,
    set_request_filter,
};

/// Serialises the tests in this binary: they all read or write the process-global filter.
fn lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[test]
fn only_the_targeted_request_records() {
    let _serialised = lock();
    assert!(request_filter().is_none());
    assert!(is_request_traced(), "no filter means every thread records");

    bind_current_request(Some("req-a".to_owned()));
    assert_eq!(set_request_filter(Some("req-b".to_owned())), None);
    assert_eq!(request_filter().as_deref(), Some("req-b"));
    assert!(
        !is_request_traced(),
        "req-a must not be recorded while req-b is targeted"
    );

    bind_current_request(Some("req-b".to_owned()));
    assert!(is_request_traced());

    // A thread that bound nothing is not traced either, so background work cannot leak into a
    // single-request capture.
    let unbound = std::thread::spawn(is_request_traced)
        .join()
        .expect("thread");
    assert!(!unbound);

    // The previous filter is returned so a caller can restore it.
    assert_eq!(set_request_filter(None).as_deref(), Some("req-b"));
    assert!(is_request_traced());
    assert!(request_filter().is_none());

    bind_current_request(None);
    assert_eq!(active_mask(), TraceScope::NONE);
}

/// A new capture must never inherit an old filter: the header-driven path arms without a
/// request id, so if arming left the filter installed it would record nothing at all.
#[test]
fn arming_clears_a_filter_left_by_an_earlier_capture() {
    let _serialised = lock();
    assert_eq!(set_request_filter(Some("req-old".to_owned())), None);
    assert!(!is_request_traced(), "no thread is bound yet");

    arm(TraceScope::ENGRAM_PREFETCH, 4);
    assert!(
        request_filter().is_none(),
        "arm() must clear a stale request filter"
    );
    assert!(is_request_traced());

    // And so must an explicit disarm, even if a filter was installed afterwards.
    set_request_filter(Some("req-next".to_owned()));
    assert!(!is_request_traced());
    disarm();
    assert!(request_filter().is_none());
    assert!(is_request_traced());
    assert_eq!(active_mask(), TraceScope::NONE);
}
