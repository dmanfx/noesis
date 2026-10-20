import React from 'react';
import '../styles/choice-strip.css';

export type ChoiceStripProps = {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
  compact?: boolean;
};

/** Direct choices retain native button activation and visible keyboard focus. */
export function ChoiceStrip({ label, value, options, onChange, compact = false }: ChoiceStripProps) {
  return <div className={`choice-strip${compact ? ' choice-strip--compact' : ''}`} role="group" aria-label={label}>
    <span className="choice-strip__label" aria-hidden="true">{label}</span>
    <div className="choice-strip__options">
      {options.map(option => <button key={option.value} type="button" aria-pressed={value === option.value}
        className={value === option.value ? 'is-selected' : ''} title={option.label}
        onClick={() => onChange(option.value)}>{option.label}</button>)}
    </div>
  </div>;
}
