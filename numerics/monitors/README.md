<!--
Copyright (c) 2026 Hunter L. Allen

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Monitor dumps

The first real reference/candidate pair goes here, with the command in docs/NUMERICS.md §6:

```bash
python3 scripts/numerics.py monitors --reference numerics/monitors/<model>-reference.json \
    --candidate numerics/monitors/<model>-candidate.json --layer-bar X --expert-bar Y
```

Nothing is committed yet, so `[prerequisites] monitors` stays false. Do not add a dump that was
not produced by a serving run; a synthetic fixture belongs in the self-tests, not here.
