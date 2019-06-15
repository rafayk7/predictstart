import React from 'react';
import logo from './logo.svg';
import './styles/App.css';
import AddUrl from './components/AddUrl';
import ToolBar from './components/ToolBar';
import logoImg from './images/logoimgimproved.png';
import AboutMe from './components/AboutMe'
import HowWorks from './components/HowWorks'

import {
  Route,
  NavLink,
  HashRouter
} from "react-router-dom";

function App() {
  return (
    <HashRouter>
      <div>
        <header>
          <ToolBar />
        </header>
      </div>
      <div className="center">
        <Route exact path="/" component={AddUrl} />
        <Route path="/aboutMe" component={AboutMe} />
        <Route path="/howWorks" component={HowWorks} />
      </div>
    </HashRouter>
  );
}

export default App;
