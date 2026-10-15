//! Dual-rail device discovery, validation and selection.
//!
//! Spec section 3 and checkpoint 6.1 name the rails by their f0 functions
//! (SPEC_RAIL0_INTERFACE and SPEC_RAIL1_INTERFACE). Measured read-only on the live four-node
//! cluster those functions are DOWN at 40 Gb/s; the ACTIVE 200 Gb/s rails are the f1 functions of
//! the same two cards. Registering a device by name without checking its port state would
//! therefore either fail outright or silently pin the milestone's bandwidth claims to a 40 Gb/s
//! link, which is why this module never hardcodes a device:
//!
//! * it enumerates the RDMA devices (FabricView::from_sysfs, or an injected view in tests),
//! * it keeps only ports whose sysfs state is ACTIVE and whose negotiated rate is at least
//!   MIN_RAIL_RATE_GBPS,
//! * it prefers the spec-declared card order, so a host whose f0 functions are healthy still gets
//!   the spec rails, and marks any deviation (SelectedRail::deviates_from_spec),
//! * it lets an operator pin the pair explicitly through RailOverride, which is validated against
//!   the same state, rate and one-device-per-card rules, so an override cannot produce a pair that
//!   discovery would refuse,
//! * and it fails loudly with the full device report when fewer than two usable rails exist, or
//!   when a pinned rail is DOWN or slower than 200 Gb/s.
//!
//! Discovery takes the sysfs root as an argument and keeps the raw state and rate text, so the
//! parsing and the selection are unit tested against a synthetic sysfs tree. Nothing in the
//! default-feature test suite reads the live fabric: the only paths that do are the rdma-feature
//! startup entry point, select_rails_from_live_fabric, which a host with an RDMA subsystem reaches
//! through RdmaTransport::open, and the ignored hardware gates that exercise it.

use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::CommError;

/// Rail 0 device name declared by the spec (ConnectX-7 f0 function, PCIe root 0).
///
/// On the live nodes this function is DOWN at 40 Gb/s; it is documentation and a preference, not
/// a registration target.
pub const SPEC_RAIL0_INTERFACE: &str = "rocep1s0f0";
/// Rail 1 device name declared by the spec (ConnectX-7 f0 function, PCIe root 2).
///
/// On the live nodes this function is DOWN at 40 Gb/s; it is documentation and a preference, not
/// a registration target.
pub const SPEC_RAIL1_INTERFACE: &str = "roceP2p1s0f0";
/// Default sysfs root holding one directory per RDMA device.
pub const SYSFS_INFINIBAND_ROOT: &str = "/sys/class/infiniband";
/// Environment variable that pins the rails, for example SPARKFLASH_RAIL_DEVICES=rocep1s0f1,roceP2p1s0f1.
pub const RAIL_OVERRIDE_ENV: &str = "SPARKFLASH_RAIL_DEVICES";
/// Lowest negotiated rate a rail may report, in Gb/s (200GbE HDR per the spec).
pub const MIN_RAIL_RATE_GBPS: u32 = 200;
/// Number of rails the RoCEnante one-shot path drives in parallel.
pub const RAIL_COUNT: usize = 2;

/// Link state of one RDMA port, as sysfs reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceState {
    /// The port negotiated a link and is usable.
    Active,
    /// The port exists but carries no traffic.
    Down,
    /// The port state could not be read or is not one we recognize.
    Unknown,
}

impl DeviceState {
    /// Stable lowercase name for logs and telemetry payloads.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Active => "ACTIVE",
            Self::Down => "DOWN",
            Self::Unknown => "UNKNOWN",
        }
    }

    /// Whether the port is usable.
    #[must_use]
    pub fn is_active(self) -> bool {
        matches!(self, Self::Active)
    }

    /// Parses a sysfs ports/N/state value such as "4: ACTIVE" or "1: DOWN".
    #[must_use]
    pub fn from_sysfs(text: &str) -> Self {
        let trimmed = text.trim();
        let name = trimmed.rsplit(':').next().unwrap_or("").trim();
        match name {
            "ACTIVE" => Self::Active,
            "DOWN" | "INIT" | "INITIALIZE" | "ARMED" => Self::Down,
            _ => Self::Unknown,
        }
    }
}

/// Parses a sysfs ports/N/rate value such as "200 Gb/sec (4X HDR)" into Gb/s.
///
/// Text without a leading number is reported as 0, which fails the rate check loudly instead of
/// pretending the link is fast.
#[must_use]
pub fn parse_rate_gbps(text: &str) -> u32 {
    let digits: String = text
        .trim()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    digits.parse().unwrap_or(0)
}

/// One resolved RDMA device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RdmaDevice {
    /// Device name, for example rocep1s0f1.
    pub name: String,
    /// Netdev behind the device, for example enp1s0f1np1 (may be empty when sysfs is unreadable).
    pub netdev: String,
    /// Port state.
    pub state: DeviceState,
    /// Negotiated rate in Gb/s, 0 when unknown.
    pub rate_gbps: u32,
}

impl RdmaDevice {
    /// Resolves one device from raw sysfs text.
    #[must_use]
    pub fn from_sysfs(name: &str, netdev: &str, port_state: &str, port_rate: &str) -> Self {
        Self {
            name: name.to_owned(),
            netdev: netdev.to_owned(),
            state: DeviceState::from_sysfs(port_state),
            rate_gbps: parse_rate_gbps(port_rate),
        }
    }

    /// Whether this port is ACTIVE and fast enough to carry a rail.
    #[must_use]
    pub fn is_usable(&self) -> bool {
        self.state.is_active() && self.rate_gbps >= MIN_RAIL_RATE_GBPS
    }

    /// Whether this device is the one the spec names for a rail index.
    #[must_use]
    pub fn is_spec_device(&self, index: usize) -> bool {
        self.name == spec_device(index)
    }

    /// One line description used in the startup report and in error messages.
    #[must_use]
    pub fn summary(&self) -> String {
        let netdev = if self.netdev.is_empty() {
            "no-netdev".to_owned()
        } else {
            self.netdev.clone()
        };
        format!(
            "{}/{netdev} {} {}Gb/s",
            self.name,
            self.state.as_str(),
            self.rate_gbps
        )
    }
}

/// The device name the spec declares for a rail index.
#[must_use]
pub fn spec_device(index: usize) -> &'static str {
    match index {
        1 => SPEC_RAIL1_INTERFACE,
        _ => SPEC_RAIL0_INTERFACE,
    }
}

/// Everything the RDMA subsystem exposes on one host.
///
/// Tests build this from an injected list or from a synthetic sysfs tree; production code calls
/// from_sysfs with SYSFS_INFINIBAND_ROOT.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FabricView {
    devices: Vec<RdmaDevice>,
}

impl FabricView {
    /// Builds a view from already resolved devices.
    #[must_use]
    pub fn new(devices: Vec<RdmaDevice>) -> Self {
        Self { devices }
    }

    /// All devices, sorted by name.
    #[must_use]
    pub fn devices(&self) -> &[RdmaDevice] {
        &self.devices
    }

    /// Looks a device up by name.
    #[must_use]
    pub fn find(&self, name: &str) -> Option<&RdmaDevice> {
        self.devices.iter().find(|device| device.name == name)
    }

    /// Devices that are ACTIVE and at least MIN_RAIL_RATE_GBPS.
    #[must_use]
    pub fn usable_devices(&self) -> Vec<&RdmaDevice> {
        self.devices
            .iter()
            .filter(|device| device.is_usable())
            .collect()
    }

    /// One line per device, used as evidence in the loud startup errors.
    #[must_use]
    pub fn summary(&self) -> String {
        if self.devices.is_empty() {
            return "no RDMA devices found".to_owned();
        }
        self.devices
            .iter()
            .map(RdmaDevice::summary)
            .collect::<Vec<_>>()
            .join(", ")
    }

    /// Reads the state and rate of the first port of every device directory, plus the first
    /// netdev listed under that device's net directory.
    ///
    /// A device whose state or rate file is unreadable stays in the view with an unknown state so
    /// the failure report can still name it.
    ///
    /// # Errors
    /// Returns CommError::FabricUnavailable when the directory itself cannot be enumerated.
    pub fn from_sysfs(root: impl AsRef<Path>) -> Result<Self, CommError> {
        let root = root.as_ref();
        let entries = std::fs::read_dir(root).map_err(|error| {
            CommError::FabricUnavailable(format!(
                "cannot enumerate RDMA devices under {}: {error}",
                root.display()
            ))
        })?;
        let mut devices = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|error| {
                CommError::FabricUnavailable(format!(
                    "cannot read an entry of {}: {error}",
                    root.display()
                ))
            })?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = entry.path();
            let port = path.join("ports").join("1");
            let state_text = read_trimmed(&port.join("state")).unwrap_or_default();
            let rate_text = read_trimmed(&port.join("rate")).unwrap_or_default();
            let netdev = first_entry_name(&path.join("device").join("net")).unwrap_or_default();
            devices.push(RdmaDevice::from_sysfs(
                &name,
                &netdev,
                &state_text,
                &rate_text,
            ));
        }
        devices.sort_by(|left, right| left.name.cmp(&right.name));
        Ok(Self { devices })
    }
}

/// Reads a file and trims it, reporting unreadable files as None.
fn read_trimmed(path: &Path) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|text| text.trim().to_owned())
}

/// Name of the first entry of a directory, if it has one.
fn first_entry_name(path: &Path) -> Option<String> {
    let mut entries = std::fs::read_dir(path).ok()?;
    let entry = entries.next()?;
    let entry = entry.ok()?;
    Some(entry.file_name().to_string_lossy().into_owned())
}

/// An operator's explicit rail pinning.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RailOverride {
    /// Device name for rail 0.
    pub rail0: String,
    /// Device name for rail 1.
    pub rail1: String,
}

impl RailOverride {
    /// Pins the two rails by name.
    #[must_use]
    pub fn new(rail0: impl Into<String>, rail1: impl Into<String>) -> Self {
        Self {
            rail0: rail0.into(),
            rail1: rail1.into(),
        }
    }

    /// Device name for a rail index.
    #[must_use]
    pub fn device(&self, index: usize) -> &str {
        if index == 1 { &self.rail1 } else { &self.rail0 }
    }

    /// Parses the RAIL_OVERRIDE_ENV value, which must name exactly two devices.
    ///
    /// # Errors
    /// Returns CommError::FabricUnavailable when the value does not contain exactly two
    /// non-empty names.
    pub fn parse(value: &str) -> Result<Self, CommError> {
        let names: Vec<&str> = value
            .split(',')
            .map(str::trim)
            .filter(|name| !name.is_empty())
            .collect();
        match names.as_slice() {
            [rail0, rail1] => Ok(Self::new(*rail0, *rail1)),
            other => Err(CommError::FabricUnavailable(format!(
                "{RAIL_OVERRIDE_ENV} must name exactly two RDMA devices, got {}: {other:?}",
                other.len()
            ))),
        }
    }

    /// Reads the pinning from the environment, if it is set.
    ///
    /// # Errors
    /// Returns CommError::FabricUnavailable when the variable is set but malformed.
    pub fn from_env() -> Result<Option<Self>, CommError> {
        match std::env::var(RAIL_OVERRIDE_ENV) {
            Ok(value) if !value.trim().is_empty() => Self::parse(&value).map(Some),
            _ => Ok(None),
        }
    }
}

/// How the rails were chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RailSelectionSource {
    /// Chosen from the enumerated devices, preferring the spec's card order.
    Discovered,
    /// Pinned by RailOverride.
    Override,
}

impl RailSelectionSource {
    /// Stable name for logs and telemetry payloads.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Discovered => "discovered",
            Self::Override => "override",
        }
    }
}

/// One validated rail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectedRail {
    /// Rail index, 0 or 1.
    pub index: usize,
    /// Device name the spec declares for this rail.
    pub spec_device: String,
    /// The device actually selected.
    pub device: RdmaDevice,
    /// Whether the selected device differs from the spec name (true when the f1 rails are used).
    pub deviates_from_spec: bool,
}

impl SelectedRail {
    /// Selected device name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.device.name
    }

    /// Netdev behind the selected device.
    #[must_use]
    pub fn netdev(&self) -> &str {
        &self.device.netdev
    }

    /// Port state of the selected device.
    #[must_use]
    pub fn state(&self) -> DeviceState {
        self.device.state
    }

    /// Negotiated rate in Gb/s.
    #[must_use]
    pub fn rate_gbps(&self) -> u32 {
        self.device.rate_gbps
    }

    /// One line description, including the deviation from the spec name.
    #[must_use]
    pub fn summary(&self) -> String {
        let deviation = if self.deviates_from_spec {
            format!(" DEVIATES from spec {}", self.spec_device)
        } else {
            " matches spec".to_owned()
        };
        format!("rail {}: {}{deviation}", self.index, self.device.summary())
    }
}

/// The two validated rails this process will register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RailSelection {
    /// How the pair was chosen.
    pub source: RailSelectionSource,
    /// Rail 0 and rail 1.
    pub rails: [SelectedRail; 2],
}

impl RailSelection {
    /// Chooses and validates the two rails.
    ///
    /// Without an override the usable devices are ranked by the spec's card order and function
    /// index, and exactly one device is taken per card, because the two rails must sit on
    /// different PCIe roots: a healthy host keeps the spec rails, and a host whose f0 functions
    /// are DOWN falls back to the f1 functions of the same two cards. An override pins the pair
    /// explicitly, but the pinned pair is still validated for ACTIVE state, for the 200 Gb/s
    /// minimum and for the same one-device-per-card rule, so an operator typo cannot silently
    /// produce a single-rail collective. The override may not produce what discovery refuses to
    /// produce.
    ///
    /// # Errors
    /// Returns CommError::FabricUnavailable when a pinned device is missing, DOWN or too slow,
    /// when both rails resolve to the same device or to two functions of one card, or when fewer
    /// than RAIL_COUNT usable rails exist. Every message carries the full device report so a
    /// startup failure names the DOWN, 40 Gb/s or colliding device that caused it.
    pub fn resolve(view: &FabricView, pinned: Option<&RailOverride>) -> Result<Self, CommError> {
        match pinned {
            Some(pinned) => {
                let rail0 = validate_rail(0, view, pinned.device(0))?;
                let rail1 = validate_rail(1, view, pinned.device(1))?;
                // Discovery refuses to put both rails on one card, and the override must not be
                // able to smuggle that configuration in: two functions of one card share a PCIe
                // root, so they are not independent rails, and a repeated device name is the
                // same mistake spelled differently.
                let same_device = rail0.name() == rail1.name();
                if same_device || card_prefix(rail0.name()) == card_prefix(rail1.name()) {
                    let reason = if same_device {
                        "the same device twice"
                    } else {
                        "two functions of the same card, which share one PCIe root"
                    };
                    return Err(CommError::FabricUnavailable(format!(
                        "rails 0 and 1 are pinned to {} and {}: {reason}, so they are not two independent rails and discovery refuses the same pair. The host reports: {}",
                        rail0.name(),
                        rail1.name(),
                        view.summary()
                    )));
                }
                Ok(Self {
                    source: RailSelectionSource::Override,
                    rails: [rail0, rail1],
                })
            }
            None => {
                let devices = discovered_devices(view)?;
                Ok(Self {
                    source: RailSelectionSource::Discovered,
                    rails: [
                        selected_rail(0, devices[0].clone()),
                        selected_rail(1, devices[1].clone()),
                    ],
                })
            }
        }
    }

    /// Rail by index, when the index is 0 or 1.
    #[must_use]
    pub fn rail(&self, index: usize) -> Option<&SelectedRail> {
        self.rails.get(index)
    }

    /// Devices in rail order.
    #[must_use]
    pub fn names(&self) -> [&str; 2] {
        [self.rails[0].name(), self.rails[1].name()]
    }

    /// Whether either rail deviates from the name the spec declares.
    #[must_use]
    pub fn deviates_from_spec(&self) -> bool {
        self.rails.iter().any(|rail| rail.deviates_from_spec)
    }

    /// Startup report: both rails with their netdev, state and rate.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "{} via {}; {}",
            self.rails[0].summary(),
            self.source.as_str(),
            self.rails[1].summary()
        )
    }
}

impl fmt::Display for RailSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.summary())
    }
}

/// Picks one usable device per card, preferring the spec's card order and the lowest function
/// index, and fails loudly when fewer than RAIL_COUNT cards can carry a rail.
fn discovered_devices(view: &FabricView) -> Result<[RdmaDevice; RAIL_COUNT], CommError> {
    let mut usable: Vec<&RdmaDevice> = view.usable_devices();
    usable.sort_by_key(|left| selection_key(left));
    let mut chosen: Vec<RdmaDevice> = Vec::with_capacity(RAIL_COUNT);
    let mut cards: Vec<String> = Vec::with_capacity(RAIL_COUNT);
    for device in usable {
        let card = card_prefix(&device.name).to_owned();
        if cards.contains(&card) {
            continue;
        }
        cards.push(card);
        chosen.push(device.clone());
        if chosen.len() == RAIL_COUNT {
            break;
        }
    }
    if chosen.len() < RAIL_COUNT {
        return Err(CommError::FabricUnavailable(format!(
            "need {RAIL_COUNT} ACTIVE rails at {MIN_RAIL_RATE_GBPS} Gb/s or faster on distinct cards, found {} usable rail(s) on {} card(s); the host reports: {}",
            chosen.len(),
            cards.len(),
            view.summary()
        )));
    }
    match chosen.as_slice() {
        [first, second] => Ok([first.clone(), second.clone()]),
        other => Err(CommError::FabricUnavailable(format!(
            "rail selection produced {} devices on {} card(s) instead of {RAIL_COUNT}; the host reports: {}",
            other.len(),
            cards.len(),
            view.summary()
        ))),
    }
}

/// Builds a SelectedRail for a discovered device.
fn selected_rail(index: usize, device: RdmaDevice) -> SelectedRail {
    let spec_device = spec_device(index).to_owned();
    let deviates_from_spec = device.name != spec_device;
    SelectedRail {
        index,
        spec_device,
        device,
        deviates_from_spec,
    }
}

/// Looks a pinned device up and validates it.
fn validate_rail(index: usize, view: &FabricView, name: &str) -> Result<SelectedRail, CommError> {
    let device = view.find(name).ok_or_else(|| {
        CommError::FabricUnavailable(format!(
            "rail {index} is pinned to {name}, which is not present; the host reports: {}",
            view.summary()
        ))
    })?;
    if !device.state.is_active() {
        return Err(CommError::FabricUnavailable(format!(
            "rail {index} is pinned to {}, whose port is {}; refusing to register a rail that is not ACTIVE. The host reports: {}",
            device.name,
            device.state.as_str(),
            view.summary()
        )));
    }
    if device.rate_gbps < MIN_RAIL_RATE_GBPS {
        return Err(CommError::FabricUnavailable(format!(
            "rail {index} is pinned to {}, which negotiated {} Gb/s, below the {} Gb/s minimum; refusing to register a slow rail. The host reports: {}",
            device.name,
            device.rate_gbps,
            MIN_RAIL_RATE_GBPS,
            view.summary()
        )));
    }
    Ok(selected_rail(index, device.clone()))
}

/// Ranking key that keeps the spec's card order while allowing the function index to differ.
fn selection_key(device: &RdmaDevice) -> (u8, u32, String) {
    let prefix = card_prefix(&device.name);
    let card_rank = if prefix == card_prefix(SPEC_RAIL0_INTERFACE) {
        0
    } else if prefix == card_prefix(SPEC_RAIL1_INTERFACE) {
        1
    } else {
        2
    };
    (card_rank, function_index(&device.name), device.name.clone())
}

/// Device name without its trailing function index, for example rocep1s0 from rocep1s0f1.
fn card_prefix(name: &str) -> &str {
    match name.rfind('f') {
        Some(index) if name[index + 1..].chars().all(|c| c.is_ascii_digit()) => &name[..index],
        _ => name,
    }
}

/// Trailing function index of a device name, 0 when there is none.
fn function_index(name: &str) -> u32 {
    match name.rsplit_once('f') {
        Some((_, digits)) => digits.parse().unwrap_or(0),
        None => 0,
    }
}

/// Serializable per-rail device report that travels with the collective telemetry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FabricTelemetrySnapshot {
    /// How the pair was chosen.
    pub source: RailSelectionSource,
    /// Rail 0 and rail 1.
    pub rails: [RailTelemetrySnapshot; 2],
}

impl FabricTelemetrySnapshot {
    /// Copies a selection into its telemetry shape.
    #[must_use]
    pub fn from_selection(selection: &RailSelection) -> Self {
        Self {
            source: selection.source,
            rails: [
                RailTelemetrySnapshot::from_selected(&selection.rails[0]),
                RailTelemetrySnapshot::from_selected(&selection.rails[1]),
            ],
        }
    }
}

/// Serializable state, rate and speed of one rail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RailTelemetrySnapshot {
    /// Rail index, 0 or 1.
    pub index: usize,
    /// Selected device name.
    pub device: String,
    /// Netdev behind the device.
    pub netdev: String,
    /// Port state.
    pub state: DeviceState,
    /// Negotiated rate in Gb/s.
    pub rate_gbps: u32,
    /// Device name the spec declares for this rail.
    pub spec_device: String,
    /// Whether the selected device differs from the spec name.
    pub deviates_from_spec: bool,
}

impl RailTelemetrySnapshot {
    fn from_selected(rail: &SelectedRail) -> Self {
        Self {
            index: rail.index,
            device: rail.device.name.clone(),
            netdev: rail.device.netdev.clone(),
            state: rail.device.state,
            rate_gbps: rail.device.rate_gbps,
            spec_device: rail.spec_device.clone(),
            deviates_from_spec: rail.deviates_from_spec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A synthetic copy of the live cluster's four RDMA functions.
    fn live_like_view() -> FabricView {
        FabricView::new(vec![
            RdmaDevice {
                name: "roceP2p1s0f0".to_owned(),
                netdev: "enP2p1s0f0np0".to_owned(),
                state: DeviceState::Down,
                rate_gbps: 40,
            },
            RdmaDevice {
                name: "roceP2p1s0f1".to_owned(),
                netdev: "enP2p1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "rocep1s0f0".to_owned(),
                netdev: "enp1s0f0np0".to_owned(),
                state: DeviceState::Down,
                rate_gbps: 40,
            },
            RdmaDevice {
                name: "rocep1s0f1".to_owned(),
                netdev: "enp1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
        ])
    }

    #[test]
    fn parses_sysfs_state_and_rate_text() {
        assert_eq!(DeviceState::from_sysfs("4: ACTIVE"), DeviceState::Active);
        assert_eq!(DeviceState::from_sysfs("1: DOWN\n"), DeviceState::Down);
        assert_eq!(DeviceState::from_sysfs("2: INIT"), DeviceState::Down);
        assert_eq!(DeviceState::from_sysfs(""), DeviceState::Unknown);
        assert_eq!(DeviceState::from_sysfs("banana"), DeviceState::Unknown);
        assert!(DeviceState::Active.is_active());
        assert!(!DeviceState::Down.is_active());
        assert_eq!(DeviceState::Unknown.as_str(), "UNKNOWN");

        assert_eq!(parse_rate_gbps("200 Gb/sec (4X HDR)"), 200);
        assert_eq!(parse_rate_gbps(" 40 Gb/sec (4X QDR)\n"), 40);
        assert_eq!(parse_rate_gbps("unknown"), 0);
        assert_eq!(parse_rate_gbps(""), 0);
    }

    #[test]
    fn selects_the_active_200g_rails_when_the_spec_functions_are_down() {
        let view = live_like_view();
        let selection = RailSelection::resolve(&view, None)
            .unwrap_or_else(|error| panic!("resolution failed: {error}"));
        assert_eq!(selection.source, RailSelectionSource::Discovered);
        // The spec's card order decides which card is rail 0, so the f1 functions land in the
        // same order the spec would have used for the f0 functions.
        assert_eq!(selection.names(), ["rocep1s0f1", "roceP2p1s0f1"]);
        assert_eq!(
            selection.rail(0).map(SelectedRail::netdev),
            Some("enp1s0f1np1")
        );
        assert_eq!(selection.rail(0).map(SelectedRail::rate_gbps), Some(200));
        assert_eq!(
            selection.rail(0).map(SelectedRail::state),
            Some(DeviceState::Active)
        );
        assert!(selection.deviates_from_spec());
        assert_eq!(selection.rails[0].spec_device, SPEC_RAIL0_INTERFACE);
        assert_eq!(selection.rails[1].spec_device, SPEC_RAIL1_INTERFACE);
        assert!(selection.summary().contains("DEVIATES"));
        assert!(selection.summary().contains("enp1s0f1np1"));
        assert_eq!(selection.names().len(), RAIL_COUNT);
        assert!(selection.rail(2).is_none());
        // The display form is what the startup log prints.
        assert_eq!(format!("{selection}"), selection.summary());
    }

    #[test]
    fn prefers_the_spec_devices_when_they_are_healthy() {
        let view = FabricView::new(vec![
            RdmaDevice {
                name: "rocep1s0f0".to_owned(),
                netdev: "enp1s0f0np0".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "roceP2p1s0f0".to_owned(),
                netdev: "enP2p1s0f0np0".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "rocep1s0f1".to_owned(),
                netdev: "enp1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "roceP2p1s0f1".to_owned(),
                netdev: "enP2p1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
        ]);
        let selection = RailSelection::resolve(&view, None)
            .unwrap_or_else(|error| panic!("resolution failed: {error}"));
        assert_eq!(
            selection.names(),
            [SPEC_RAIL0_INTERFACE, SPEC_RAIL1_INTERFACE]
        );
        assert!(!selection.deviates_from_spec());
        assert!(selection.summary().contains("matches spec"));
    }

    #[test]
    fn an_override_is_validated_exactly_like_discovery() {
        let view = live_like_view();
        let pinned = RailOverride::new("rocep1s0f1", "roceP2p1s0f1");
        let selection = RailSelection::resolve(&view, Some(&pinned))
            .unwrap_or_else(|error| panic!("resolution failed: {error}"));
        assert_eq!(selection.source, RailSelectionSource::Override);
        assert_eq!(selection.names(), ["rocep1s0f1", "roceP2p1s0f1"]);

        // Pinning the DOWN 40 Gb/s f0 function must fail loudly, naming the state and the rate.
        let pinned = RailOverride::new("rocep1s0f0", "roceP2p1s0f1");
        match RailSelection::resolve(&view, Some(&pinned)) {
            Err(CommError::FabricUnavailable(message)) => {
                assert!(message.contains("rocep1s0f0"), "{message}");
                assert!(message.contains("DOWN"), "{message}");
                assert!(message.contains("enp1s0f0np0"), "{message}");
            }
            other => panic!("a DOWN rail must be refused, got {other:?}"),
        }

        // A present but missing device is refused too.
        let pinned = RailOverride::new("rocep9s9f9", "roceP2p1s0f1");
        assert!(matches!(
            RailSelection::resolve(&view, Some(&pinned)),
            Err(CommError::FabricUnavailable(message)) if message.contains("not present")
        ));
    }

    #[test]
    fn an_override_may_not_pin_the_same_device_twice() {
        let view = live_like_view();
        for pinned in [
            RailOverride::new("rocep1s0f1", "rocep1s0f1"),
            RailOverride::new("roceP2p1s0f1", "roceP2p1s0f1"),
        ] {
            match RailSelection::resolve(&view, Some(&pinned)) {
                Err(CommError::FabricUnavailable(message)) => {
                    assert!(message.contains("same device twice"), "{message}");
                    assert!(message.contains(&pinned.rail0), "{message}");
                    assert!(message.contains("Gb/s"), "{message}");
                }
                other => panic!("pinning one device twice must be refused, got {other:?}"),
            }
        }
    }

    #[test]
    fn an_override_may_not_pin_two_functions_of_one_card() {
        let view = FabricView::new(vec![
            RdmaDevice {
                name: "rocep1s0f1".to_owned(),
                netdev: "enp1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "rocep1s0f2".to_owned(),
                netdev: "enp1s0f2np2".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "roceP2p1s0f1".to_owned(),
                netdev: "enP2p1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
        ]);
        let pinned = RailOverride::new("rocep1s0f1", "rocep1s0f2");
        match RailSelection::resolve(&view, Some(&pinned)) {
            Err(CommError::FabricUnavailable(message)) => {
                assert!(
                    message.contains("two functions of the same card"),
                    "{message}"
                );
                assert!(message.contains("rocep1s0f1"), "{message}");
                assert!(message.contains("rocep1s0f2"), "{message}");
                assert!(message.contains("enp1s0f2np2"), "{message}");
            }
            other => panic!("two functions of one card are not two rails, got {other:?}"),
        }
        // The same view still resolves a genuine two-card pair.
        let ok = RailSelection::resolve(
            &view,
            Some(&RailOverride::new("rocep1s0f2", "roceP2p1s0f1")),
        )
        .unwrap_or_else(|error| panic!("a two-card pin must still resolve: {error}"));
        assert_eq!(ok.names(), ["rocep1s0f2", "roceP2p1s0f1"]);
    }

    #[test]
    fn refuses_an_active_but_slow_rail() {
        let view = FabricView::new(vec![
            RdmaDevice {
                name: "rocep1s0f1".to_owned(),
                netdev: "enp1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "roceP2p1s0f0".to_owned(),
                netdev: "enP2p1s0f0np0".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 40,
            },
        ]);
        // Only one usable rail: discovery refuses with the full report.
        match RailSelection::resolve(&view, None) {
            Err(CommError::FabricUnavailable(message)) => {
                assert!(message.contains("found 1"), "{message}");
                assert!(message.contains("40Gb/s"), "{message}");
            }
            other => panic!("one usable rail must be refused, got {other:?}"),
        }
        // An override onto the 40 Gb/s device is refused as slow.
        let pinned = RailOverride::new("rocep1s0f1", "roceP2p1s0f0");
        match RailSelection::resolve(&view, Some(&pinned)) {
            Err(CommError::FabricUnavailable(message)) => {
                assert!(message.contains("below the 200 Gb/s minimum"), "{message}");
                assert!(message.contains("40 Gb/s"), "{message}");
            }
            other => panic!("a 40 Gb/s rail must be refused, got {other:?}"),
        }
    }

    #[test]
    fn refuses_two_usable_rails_on_the_same_card() {
        let view = FabricView::new(vec![
            RdmaDevice {
                name: "rocep1s0f0".to_owned(),
                netdev: "enp1s0f0np0".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
            RdmaDevice {
                name: "rocep1s0f1".to_owned(),
                netdev: "enp1s0f1np1".to_owned(),
                state: DeviceState::Active,
                rate_gbps: 200,
            },
        ]);
        match RailSelection::resolve(&view, None) {
            Err(CommError::FabricUnavailable(message)) => {
                assert!(message.contains("distinct cards"), "{message}");
                assert!(message.contains("found 1"), "{message}");
            }
            other => panic!("two functions of one card are not two rails, got {other:?}"),
        }
    }

    #[test]
    fn refuses_a_host_with_no_rdma_devices() {
        let view = FabricView::default();
        assert_eq!(view.summary(), "no RDMA devices found");
        assert!(view.usable_devices().is_empty());
        assert!(matches!(
            RailSelection::resolve(&view, None),
            Err(CommError::FabricUnavailable(message)) if message.contains("found 0")
        ));
    }

    #[test]
    fn parses_the_override_environment_value() {
        let parsed = RailOverride::parse(" rocep1s0f1 , roceP2p1s0f1 ")
            .unwrap_or_else(|error| panic!("parse failed: {error}"));
        assert_eq!(parsed.rail0, "rocep1s0f1");
        assert_eq!(parsed.rail1, "roceP2p1s0f1");
        assert_eq!(parsed.device(0), "rocep1s0f1");
        assert_eq!(parsed.device(1), "roceP2p1s0f1");
        assert!(matches!(
            RailOverride::parse("rocep1s0f1"),
            Err(CommError::FabricUnavailable(_))
        ));
        assert!(matches!(
            RailOverride::parse("a,b,c"),
            Err(CommError::FabricUnavailable(_))
        ));
        assert!(matches!(
            RailOverride::parse(" , "),
            Err(CommError::FabricUnavailable(_))
        ));
    }

    #[test]
    fn helper_parsers_are_stable() {
        assert_eq!(card_prefix("rocep1s0f1"), "rocep1s0");
        assert_eq!(card_prefix("roceP2p1s0f0"), "roceP2p1s0");
        assert_eq!(card_prefix("mlx5_0"), "mlx5_0");
        assert_eq!(function_index("rocep1s0f0"), 0);
        assert_eq!(function_index("rocep1s0f1"), 1);
        assert_eq!(function_index("mlx5_0"), 0);
        assert_eq!(spec_device(0), SPEC_RAIL0_INTERFACE);
        assert_eq!(spec_device(1), SPEC_RAIL1_INTERFACE);
    }

    /// Removes a synthetic sysfs tree when the test ends.
    struct TempFabric {
        root: std::path::PathBuf,
    }

    impl TempFabric {
        fn new(tag: &str) -> Self {
            let root = std::env::temp_dir()
                .join(format!("sparkflash-fabric-{}-{tag}", std::process::id()));
            let _ = std::fs::remove_dir_all(&root);
            Self { root }
        }

        fn add(&self, device: &str, netdev: &str, state: &str, rate: &str) {
            let port = self.root.join(device).join("ports").join("1");
            std::fs::create_dir_all(&port).unwrap_or_else(|error| panic!("{error}"));
            std::fs::write(port.join("state"), format!("{state}\n"))
                .unwrap_or_else(|error| panic!("{error}"));
            std::fs::write(port.join("rate"), format!("{rate}\n"))
                .unwrap_or_else(|error| panic!("{error}"));
            std::fs::create_dir_all(
                self.root
                    .join(device)
                    .join("device")
                    .join("net")
                    .join(netdev),
            )
            .unwrap_or_else(|error| panic!("{error}"));
        }
    }

    impl Drop for TempFabric {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    #[test]
    fn reads_a_synthetic_sysfs_tree_and_selects_from_it() {
        let tree = TempFabric::new("select");
        tree.add("rocep1s0f0", "enp1s0f0np0", "1: DOWN", "40 Gb/sec (4X QDR)");
        tree.add(
            "rocep1s0f1",
            "enp1s0f1np1",
            "4: ACTIVE",
            "200 Gb/sec (4X HDR)",
        );
        tree.add(
            "roceP2p1s0f0",
            "enP2p1s0f0np0",
            "1: DOWN",
            "40 Gb/sec (4X QDR)",
        );
        tree.add(
            "roceP2p1s0f1",
            "enP2p1s0f1np1",
            "4: ACTIVE",
            "200 Gb/sec (4X HDR)",
        );

        let view = FabricView::from_sysfs(&tree.root)
            .unwrap_or_else(|error| panic!("sysfs read failed: {error}"));
        assert_eq!(view.devices().len(), 4);
        // Sorted by name: roceP2p1s0f0, roceP2p1s0f1, rocep1s0f0, rocep1s0f1.
        assert_eq!(view.devices()[0].name, "roceP2p1s0f0");
        assert_eq!(view.devices()[1].state, DeviceState::Active);
        assert_eq!(view.devices()[1].rate_gbps, 200);
        assert_eq!(view.devices()[1].netdev, "enP2p1s0f1np1");
        assert_eq!(view.find("rocep1s0f0").map(|d| d.rate_gbps), Some(40));
        assert_eq!(view.find("nope"), None);

        let selection = RailSelection::resolve(&view, None)
            .unwrap_or_else(|error| panic!("resolution failed: {error}"));
        assert_eq!(selection.names(), ["rocep1s0f1", "roceP2p1s0f1"]);
        assert!(selection.deviates_from_spec());
        assert_eq!(selection.rails[1].netdev(), "enP2p1s0f1np1");
    }

    #[test]
    fn a_missing_sysfs_root_is_a_loud_error() {
        let missing = std::env::temp_dir().join("sparkflash-fabric-does-not-exist");
        match FabricView::from_sysfs(&missing) {
            Err(CommError::FabricUnavailable(message)) => {
                assert!(message.contains("cannot enumerate"), "{message}");
            }
            other => panic!("a missing sysfs root must fail, got {other:?}"),
        }
    }

    #[test]
    fn telemetry_snapshot_carries_the_device_state_and_rate() {
        let view = live_like_view();
        let selection = RailSelection::resolve(&view, None)
            .unwrap_or_else(|error| panic!("resolution failed: {error}"));
        let snapshot = FabricTelemetrySnapshot::from_selection(&selection);
        assert_eq!(snapshot.source, RailSelectionSource::Discovered);
        assert_eq!(snapshot.rails[0].device, "rocep1s0f1");
        assert_eq!(snapshot.rails[0].state, DeviceState::Active);
        assert_eq!(snapshot.rails[0].rate_gbps, 200);
        assert_eq!(snapshot.rails[0].spec_device, SPEC_RAIL0_INTERFACE);
        assert!(snapshot.rails[0].deviates_from_spec);
        assert_eq!(snapshot.rails[1].device, "roceP2p1s0f1");
        assert_eq!(snapshot.rails[1].netdev, "enP2p1s0f1np1");
    }
}
