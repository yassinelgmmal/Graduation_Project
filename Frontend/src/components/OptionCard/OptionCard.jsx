import React from 'react';
import './OptionCard.css';

const OptionCard = ({ title, description, onSelect, selected }) => {
  return (
    <div
      className={`option-card p-3 my-3 border rounded shadow-sm ${selected ? 'selected' : ''}`}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => { if (e.key === 'Enter') onSelect(); }}
      aria-pressed={selected}
      aria-label={`Select option: ${title}`}
    >
      <h5>{title}</h5>
      <p>{description}</p>
    </div>
  );
};

export default OptionCard;
