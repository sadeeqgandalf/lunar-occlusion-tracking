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
// stripe colours: hues spread round the colour wheel and kept away from the gold visor, so a camera can tell people
// apart by colour (measured with PyTorch3D renders: same person 0.99 similarity, different people ~0.1)
export const STRIPES = [
  { name: 'red', hex: '#e52929' }, { name: 'blue', hex: '#2966eb' }, { name: 'green', hex: '#29b847' }, { name: 'magenta', hex: '#e033b8' },
  { name: 'cyan', hex: '#1fbdcc' }, { name: 'violet', hex: '#8542e0' }, { name: 'lime', hex: '#99db24' }, { name: 'pink', hex: '#f2739e' },
  { name: 'indigo', hex: '#4d4da6' }, { name: 'maroon', hex: '#b31f4d' },
];
export const stripeOf = (personId) => STRIPES[(personId - 1) % STRIPES.length];
export const personName = (personId) => `P${personId}`;
