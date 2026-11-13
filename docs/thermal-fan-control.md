# Fan control on the GB10 cluster

The four Sparks (sg1-oneill .80, sg1-jackson .68, sg1-teilc .88, sg1-carter .63)
run in a rack, so fan noise is free and sustained clocks matter more than silence.
This page records what actually controls the fans, what does not, and the maximum-RPM
setup installed on all four nodes on 2026-11-13.

## What is on the machine

| Interface | Present | Controls the fan? |
| :-- | :-- | :-- |
| `ipmitool` / `/dev/ipmi0` | binary installed, no device | No BMC on a Spark; nothing to talk to |
| `nvidia-smi` fan query | yes, `fan.speed = N/A` | No; the GB10 fan is outside the GPU driver |
| `hwmon` (`/sys/class/hwmon`) | acpitz, nvme, four mlx5, mt7925 | No `pwm*` or `fan*_input` at all |
| ACPI fan cooling devices | only CPU `Processor` states | No; on newer BIOS `FAN0/FAN1` exist but the EC ignores them |
| EC thermal mailbox | yes, via FF-A eSPI partition `arm-ffa-17` | **Yes** — this is the control path |

So the fans are owned by the embedded controller. The supported knob is the EC's
"high override" slot, written over the SoC's FF-A eSPI service. On EC
`0x03000508` (what all four nodes run) the EC accepts a floor of 13500 RPM and
both fan policies saturate at 100% PWM.

## The tool

[`Z841973620/dgx-spark-fan-override`](https://github.com/Z841973620/dgx-spark-fan-override)
— a small DKMS module plus a CLI and a systemd unit.

* `/sys/bus/arm_ffa/devices/arm-ffa-17/fan` reads `ready`/`max`/`auto`/`error`.
* Writing `max` sends `07 05 00 BC 34` (command 5, 0x34BC = 13500 RPM);
  `auto` sends `07 05 00 FF FF` (slot disabled, NVIDIA curve).
* It writes EC SRAM only — no flash, no NVRAM — so it is volatile across power cycles,
  which is why the unit re-applies it at boot.
* Builds clean against `7.0.0-1019-nvidia` and signs with the local MOK (Secure Boot
  is disabled on these nodes, so no enrollment was needed).

Installed layout: source in `/usr/src/nvfancontrol`, DKMS package `nvfancontrol/1.0.0`,
CLI `/usr/sbin/nvfancontrol`, unit `/etc/systemd/system/nvfancontrol.service`
(`Type=oneshot`, `RemainAfterExit=yes`, `ExecStart=/usr/sbin/nvfancontrol max`).

## Operating it

```sh
sudo nvfancontrol status     # Module: loaded / Driver: ... / Fan: max
sudo nvfancontrol auto       # hand control back to NVIDIA's curve
sudo nvfancontrol max        # back to full speed
systemctl status nvfancontrol
```

The EC acknowledges each write in `dmesg`:

```
nvfancontrol arm-ffa-17: command-17 response: status=0 elapsed=361 us
nvfancontrol arm-ffa-17: MAX ACKNOWLEDGED: EC accepted high override=13500 RPM;
                                  fan0 and fan1 policy outputs saturate at 100%
```

`max` is a **floor**, not a clamp: the EC curve can still ask for more, and `auto`
restores the stock curve. The setting lives in EC SRAM, so it survives module unload
but not a power cycle — the boot unit covers that.

## Measured effect on sg1-oneill

Idle, 25 °C room:

| | stock curve | `max` |
| :-- | --: | --: |
| GPU | 66 °C | 47 °C after 90 s |
| Hottest ACPI zone | 74.8 °C | 50 °C after 90 s |

Sustained inference (6 concurrent 900-token completions, TP4, ~34 s each):

| Node | GPU peak | Hottest zone peak | SM clock | Throttle flags |
| :-- | --: | --: | --: | :-- |
| sg1-oneill | 58 °C | 68.0 °C | 2528 MHz | none |
| sg1-jackson | 58 °C | 69.6 °C | 2522 MHz | none |
| sg1-teilc | 58 °C | 66.2 °C | 2535 MHz | none |
| sg1-carter | 57 °C | 68.1 °C | 2483 MHz | none |

After the load ends all four fall back to ~44 °C. Before the change the same box sat
at 66 °C idle and had accumulated `SW Thermal Slowdown` and `HW Thermal Slowdown`
counter time; during the test no throttle reason was active and no new time accrued.

## Cluster rollout

All four nodes are identical (EC `0x03000508`, kernel `7.0.0-1019-nvidia`, Secure
Boot off), so the same install was applied to each. TP4 is only as fast as its
hottest rank, so cooling one node is not enough.

On a new node:

```sh
git clone --depth 1 https://github.com/Z841973620/dgx-spark-fan-override
cd dgx-spark-fan-override/nvfancontrol
sudo install -d /usr/src/nvfancontrol
sudo install -m644 usr/src/nvfancontrol/{nvfancontrol.c,Makefile,dkms.conf} /usr/src/nvfancontrol/
sudo ln -sfn /usr/src/nvfancontrol /usr/src/nvfancontrol-1.0.0
sudo install -m755 usr/sbin/nvfancontrol /usr/sbin/nvfancontrol
sudo install -m644 usr/lib/systemd/system/nvfancontrol.service /etc/systemd/system/
sudo dkms add -m nvfancontrol -v 1.0.0
sudo dkms build -m nvfancontrol -v 1.0.0 -k "$(uname -r)"
sudo dkms install -m nvfancontrol -v 1.0.0 -k "$(uname -r)"
sudo systemctl daemon-reload && sudo systemctl enable --now nvfancontrol
```

## Caveats

* **Firmware-dependent.** On EC `0x03000d07` the same command is answered
  "unsupported" (NVIDIA forum reports), so an EC update silently disables the
  override. Check `fwupdmgr get-devices | grep -A3 'Embedded Controller'` and
  re-test `sudo nvfancontrol max` after any OTA. No update was pending on these
  nodes when this was installed.
* **No RPM telemetry** through this module; the proof is the temperature drop and
  the EC acknowledgement in `dmesg`. `christopherowen/dgx-spark-fan-control`
  exposes two hwmon RPM sensors and 12 RPM floors if telemetry is wanted, but it
  binds the same FF-A partition and must not run alongside this one.
* **Full speed is loud and wears the fans.** 9000 RPM (fan0) / 13500 RPM (fan1)
  continuously is the trade the rack makes. `nvfancontrol auto` is the revert.
* **Volatile.** A hard power loss leaves the EC at its default until the unit runs
  again; nothing in flash is modified, so recovery never needs a reflash.
