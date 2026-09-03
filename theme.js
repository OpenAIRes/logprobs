/* Light / dark, resolved before the first paint.

   Loaded from <head> on every page, synchronously and on purpose: setting
   data-theme after the body has rendered shows the wrong palette for a frame,
   which reads as a flicker on every navigation -- and the bar navigates on every
   change of a setting.

   Two states. The system setting decides the first visit and nothing after
   that: a third `system` state is defensible but it makes the button say three
   things, and this is a one-button control on a strip that is meant to be
   glanceable.

   The value lives in localStorage rather than the URL: it is about the person
   reading, not about what is on screen, so a shared link should not carry it and
   force a stranger into someone else's palette. Every other setting the bar owns
   is in the URL for exactly the opposite reason.
*/
(() => {
  const KEY = 'app_theme';
  const ORDER = ['light', 'dark'];
  const query = window.matchMedia ? matchMedia('(prefers-color-scheme: dark)') : null;

  // No stored choice yet: take the machine's, once. From then on it is explicit.
  let choice = query && query.matches ? 'dark' : 'light';
  try {
    const v = localStorage.getItem(KEY);
    if (ORDER.includes(v)) choice = v;
  } catch { /* private mode: the machine's setting stands for this session */ }

  function resolved() { return choice; }

  function apply() {
    // Always explicit, so the CSS never has to guess and no rule depends on the
    // absence of an attribute.
    document.documentElement.dataset.theme = resolved();
    window.dispatchEvent(new CustomEvent('themechange',
      { detail: { choice, resolved: resolved() } }));
  }

  function set(next) {
    choice = ORDER.includes(next) ? next : 'light';
    try { localStorage.setItem(KEY, choice); } catch {}
    apply();
  }

  window.appTheme = {
    get: () => choice,
    resolved,
    set,
    toggle: () => set(choice === 'dark' ? 'light' : 'dark'),
    cycle: () => set(choice === 'dark' ? 'light' : 'dark'),
    ORDER,
  };

  apply();
})();
