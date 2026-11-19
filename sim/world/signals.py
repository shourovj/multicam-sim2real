"""Traffic signals: fixed-time plans (pure numpy).

  signals:
    - id: tl_NS
      lanes: [north_in, south_in]       one head per lane, placed by the layout (far side, over the lane)
      plan: [{state: green, until_s: 3.0}, {state: yellow, until_s: 4.0}, {state: red, until_s: 7.2}]
"""

STATES = ("red", "yellow", "green")


class Signal:
    def __init__(self, spec):
        self.id, self.lanes, self.plan = spec["id"], list(spec["lanes"]), spec["plan"]
        assert all(p["state"] in STATES for p in self.plan), f"{self.id}: states are {STATES}"
        assert all(a["until_s"] < b["until_s"] for a, b in zip(self.plan, self.plan[1:])), f"{self.id}: plan out of order"

    def state(self, t):
        for p in self.plan:
            if t < p["until_s"]:
                return p["state"]
        return self.plan[-1]["state"]

    def changes(self):
        """[(t, new_state)] for every change after t = 0."""
        return [(a["until_s"], b["state"]) for a, b in zip(self.plan, self.plan[1:])]
