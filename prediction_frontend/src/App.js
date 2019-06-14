import React from 'react';
import logo from './logo.svg';
import './styles/App.css';
import AddUrl from './components/AddUrl';
import ToolBar from './components/ToolBar';
import logoImg from './images/logoimgimproved.png';

function App() {
  return (
    <div>
      <header className="App">
        <ToolBar />
      </header>
      <AddUrl />
    </div>
    
  );
}

export default App;
