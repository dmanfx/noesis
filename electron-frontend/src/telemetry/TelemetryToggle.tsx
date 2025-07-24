import React from 'react';
import styled from 'styled-components';

const Button = styled.button`
  position: fixed;
  top: 50%;
  right: 0;
  transform: translate(50%, -50%);
  background: #2b2b2b;
  border-radius: 4px 0 0 4px;
  border: 1px solid #444;
  color: #fff;
  padding: 8px 10px;
  cursor: pointer;
  z-index: 1001;
  &:hover {
    background: #383838;
  }
`;

interface ToggleProps {
  onToggle: () => void;
}

export const TelemetryToggle: React.FC<ToggleProps> = ({ onToggle }) => (
  <Button onClick={onToggle}>
    ≡
  </Button>
);
