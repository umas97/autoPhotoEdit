// autoPhotoEdit -- automatic post-production for Sony RAW files.
// SPDX-License-Identifier: GPL-3.0-or-later
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

/** Compose class names, letting the later Tailwind class win. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs))
}

/** A number for a slider read-out: fixed decimals, no exponent, no surprises. */
export function formatNumber(value: number, digits = 2, unit = ''): string {
  return `${value.toFixed(digits).replace('.', ',')}${unit}`
}

const relative = new Intl.RelativeTimeFormat('it', { numeric: 'auto' })
const absolute = new Intl.DateTimeFormat('it', { dateStyle: 'medium', timeStyle: 'short' })

/** "3 minuti fa" for the recent past, a date for anything older than a week. */
export function formatWhen(iso: string | null): string {
  if (!iso) return ''
  const then = new Date(iso)
  const seconds = (then.getTime() - Date.now()) / 1000
  const units: Array<[Intl.RelativeTimeFormatUnit, number]> = [
    ['second', 60],
    ['minute', 3600],
    ['hour', 86400],
    ['day', 604800],
  ]
  let divisor = 1
  for (const [unit, limit] of units) {
    if (Math.abs(seconds) < limit) return relative.format(Math.round(seconds / divisor), unit)
    divisor = limit
  }
  return absolute.format(then)
}

/** Shutter speeds are read as fractions above a second, as on the camera. */
export function formatShutter(seconds: number | null): string {
  if (!seconds) return ''
  if (seconds >= 1) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}`
  return `1/${Math.round(1 / seconds)}`
}

/** A size in the units a person reads: KB under a megabyte, then MB, then GB. */
export function formatBytes(bytes: number): string {
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}
