# Daily cycling limit as an operating proxy

Some storage-system contracts specify a daily duty limit, but there is no
universal physical rule that all batteries must stop after two cycles. A
[TESVOLT LFP product announcement](https://www.tesvolt.com/en/press/tesvolt-presents-new-economic-miracle-battery-storage-system-with-high-temperature-cells.html)
describes a product-specific guarantee at two full cycles per day with its
trading software. A [Huawei LUNA2000 2 MWh warranty](https://www.sig.energy/wp-content/uploads/2024/12/luna2000-2.0mwh-warranty.pdf)
uses two daily charges and discharges in one regional baseline and a different
duty in another. A 2026 [Pramac C&I and container-system warranty](https://www.pramac.com/file/117215)
lists a maximum of two cycles per day for specified systems and operating
conditions. These are contract/product conditions, not a general safety bound.

The study's optional `battery.daily_cycle_limit_efc` is an explicit operating
assumption. It defines equivalent full cycles for each Europe/Rome calendar
day using AC-side energy and the capacity committed for that day:

```
EFC = sum((charge_MW + discharge_MW) * hours) / (2 * available_capacity_MWh)
```

This is a linear throughput proxy: it counts half of a nominal cycle for a
one-way AC transfer equal to the day's available capacity. Two such EFC permit
AC-side charge-plus-discharge energy of four times that capacity. The capacity
is fixed within each day and is updated at midnight in the SOH model; it is
nominal and fixed in the no-SOH baseline. This differs from a count of
completed rainflow cycles, a discharge-only warranty measurement, or a rolling
24-hour window. The SOH
model separately computes wear from internal energy using the directional
efficiencies; the operational cap does not change its physical model. A daily
cap alone does not enforce contract-specific annual, lifetime, temperature,
C-rate, depth or SOC conditions. Check a signed warranty before using it as a
compliance rule.

The optimizer accounts for actual already-used daily throughput in a residual
solve. The physical controller must respect the remaining budget after feedback
and protection; doing so can create imbalance against fixed market commitments.
Calendar days use elapsed time and handle 23- and 25-hour DST days. The
selected paired study applies the same two-EFC proxy to both health models.
