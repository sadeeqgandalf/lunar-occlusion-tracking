// Two colour codes with ONE meaning each, used by every view:
//  * idColor(trackId): the identity a TRACKER assigned (mask / box / tag).
//  * stripeOf(personId): who the person REALLY is, shown as coloured suit stripes. NASA marks EVA suits with stripes so
//    cameras and mission control can tell crew apart (red commander stripes since Apollo 13).
//
// Track colours come from a fixed high-contrast palette (bright subset of Kelly's colours of maximum contrast) and are
// handed out so that IDs on screen together never share a colour. An ID keeps its colour for its whole life.
const PALETTE = ['#F3C300', '#A1CAF1', '#F38400', '#E68FAC', '#00A86B', '#C2B280', '#BE0032', '#8DB600', '#0067A5', '#F99379',
  '#B3446C', '#DCD300', '#875692', '#E25822', '#2EC4B6', '#9B5DE5'];
const assigned = new Map(), recent = [];
export function idColor(id) {
  if (!assigned.has(id)) {
    const busy = new Set(recent.map((r) => assigned.get(r)));
    const pick = PALETTE.find((c) => !busy.has(c)) ?? PALETTE[id % PALETTE.length];
    assigned.set(id, pick); recent.push(id); if (recent.length > PALETTE.length - 2) recent.shift();
  }
  return assigned.get(id);
}
export function idColorA(id, a) {
  const h = idColor(id), n = parseInt(h.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}
export const STRIPES = [
  { name: 'red', hex: '#e5322d' }, { name: 'gold', hex: '#f2c12e' }, { name: 'blue', hex: '#2f6fe0' }, { name: 'green', hex: '#2fb34a' },
  { name: 'orange', hex: '#f07a1a' }, { name: 'violet', hex: '#8e4fd6' }, { name: 'cyan', hex: '#1fb5c9' }, { name: 'pink', hex: '#e64fa3' },
  { name: 'lime', hex: '#9ccc2b' }, { name: 'brown', hex: '#8a5a32' },
];
export const stripeOf = (personId) => STRIPES[(personId - 1) % STRIPES.length];
