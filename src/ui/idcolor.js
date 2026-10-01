// Two codes with ONE meaning each, used by every view:
//  * People P1–P9: who the person REALLY is, shown as coloured suit stripes. NASA marks EVA suits with stripes so
//    cameras and mission control can tell crew apart (red commander stripes since Apollo 13).
//  * Tracker IDs #11, #12, …: the identity a TRACKER assigned (mask / box / tag). Each tracker numbers its confirmed
//    tracks in order from 11, so a person who keeps one ID keeps one number, and every ID switch shows as a new number.
//
// Track colours come from a fixed high-contrast palette (Kelly's colours of maximum contrast, dark ones lightened so the
// dark chip text stays readable), indexed by the ID number: consecutive IDs, usually on screen together, never share one.
const PALETTE = ['#F3C300', '#A1CAF1', '#F38400', '#E68FAC', '#00C27C', '#C2B280', '#FF5C6C', '#8DB600', '#4FA3E0', '#F99379',
  '#D97BA4', '#DCD300', '#B98AD4', '#E25822', '#2EC4B6', '#B794F6'];
export const FIRST_ID = 11;

/** Display number of a tracker's internal track id: 11, 12, 13… in the order the tracker first confirms them. */
export function label(tracker, id) {
  const m = (tracker._labels ??= new Map());
  if (!m.has(id)) m.set(id, FIRST_ID + m.size);
  return m.get(id);
}
/** Number every confirmed track of a tracker (call once per step so numbering follows confirmation order). */
export function labelConfirmed(tracker) { for (const t of tracker.tracks) if (t.confirmed) label(tracker, t.id); }

/** Colour of a display ID (a number from label()). */
export const idColor = (n) => PALETTE[(((n - FIRST_ID) % PALETTE.length) + PALETTE.length) % PALETTE.length];
export function idColorA(n, a) {
  const h = idColor(n), v = parseInt(h.slice(1), 16);
  return `rgba(${(v >> 16) & 255}, ${(v >> 8) & 255}, ${v & 255}, ${a})`;
}
export const STRIPES = [
  { name: 'red', hex: '#e5322d' }, { name: 'gold', hex: '#f2c12e' }, { name: 'blue', hex: '#2f6fe0' }, { name: 'green', hex: '#2fb34a' },
  { name: 'orange', hex: '#f07a1a' }, { name: 'violet', hex: '#8e4fd6' }, { name: 'cyan', hex: '#1fb5c9' }, { name: 'pink', hex: '#e64fa3' },
  { name: 'lime', hex: '#9ccc2b' }, { name: 'brown', hex: '#8a5a32' },
];
export const stripeOf = (personId) => STRIPES[(personId - 1) % STRIPES.length];
export const personName = (personId) => `P${personId}`;
