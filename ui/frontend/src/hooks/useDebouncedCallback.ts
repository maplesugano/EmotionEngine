import { useEffect, useRef } from "react";

/** Returns a debounced version of `fn` that resets on every call. */
export function useDebouncedCallback<T extends (...args: never[]) => void>(
  fn: T,
  delay: number,
): (...args: Parameters<T>) => void {
  const timer = useRef<number | null>(null);
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  }, [fn]);

  useEffect(() => {
    return () => {
      if (timer.current !== null) window.clearTimeout(timer.current);
    };
  }, []);

  return (...args: Parameters<T>) => {
    if (timer.current !== null) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => {
      fnRef.current(...args);
    }, delay);
  };
}
