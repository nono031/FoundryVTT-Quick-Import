# Changelog

## 1.6.1

- **Fix (v14):** the Foundry version check `Number(game.version?.[0] ?? 0) >= 14`
  compared the first *character* of the version string (`"1"` → `1`), so the v14
  code path never ran. Scenes created on v14 received legacy fields only, leaving
  the background (Level) and ambience (`environment`) unset — the map was
  invisible after import and had to be repaired manually.
- **Fix (v14):** `ensureLevelBackground` now polls briefly (up to 3 s) for the
  embedded default Level after `Scene.create()` instead of giving up when it is
  not attached yet, and logs an error to the console when it cannot apply the
  background.

## 1.6.0

- Port to Foundry VTT v14 (compatibility minimum 12, verified on 14):
  background stored on the embedded Level, lighting on `environment`,
  grid offset via top-level `shiftX`/`shiftY`, `foundry.utils.duplicate`,
  `game.scenes.viewScene`.
