/* Light / dark, resolved before the first paint.

   Loaded from <head> on every page, synchronously and on purpose: setting
   data-theme after the body has rendered shows the wrong palette for a frame,
   which reads as a flicker on every navigation -- and the bar navigates on every
   change of a setting.

   Three states, not two. `system` follows the OS and keeps following it while
   it is chosen; `light` and `dark` override it. The distinction matters because
   "the same as my machine" is a real preference, and a two-state toggle would
   silently pin whatever the machine happened to say the first time.

   The value lives in localStorage rather than the URL: it is about the person
   reading, not about what is on screen, so a shared link should not carry it and
   force a stranger into someone else's palette. Every other setting the bar owns
   is in the URL for exactly the opposite reason.
*/
(() => {
  const KEY = 'app_theme';
  const ORDER = ['system', 'light', 'dark'];
  const query = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;

  let choice = 'system';
  try {
    const v = localStorage.getItem(KEY);
    if (ORDER.includes(v)) choice = v;
  } catch { /* private mode: fall back to following the system */ }

  function resolved() {
    if (choice !== 'system') return choice;
    return query && query.matches ? 'dark' : 'light';
  }

  function apply() {
    // Always explicit, so the CSS never has to guess and no rule depends on the
    // absence of an attribute.
    document.documentElement.dataset.theme = resolved();
    window.dispatchEvent(new CustomEvent('themechange',
      { detail: { choice, resolved: resolved() } }));
  }

  function set(next) {
    choice = ORDER.includes(next) ? next : 'system';
    try { localStorage.setItem(KEY, choice); } catch {}
    apply();
  }

  // Following the system means following it as it changes, not just at load.
  if (query && query.addEventListener) {
    query.addEventListener('change', () => { if (choice === 'system') apply(); });
  }

  window.appTheme = {
    get: () => choice,
    resolved,
    set,
    cycle: () => set(ORDER[(ORDER.indexOf(choice) + 1) % ORDER.length]),
    ORDER,
  };

  apply();
})();
