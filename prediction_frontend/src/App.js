import React from 'react';
import './styles/App.css';

import AddUrl from './components/AddUrl';
import ToolBar from './components/ToolBar';
import AboutMe from './components/AboutMe';
import HowWorks from './components/HowWorks';
import ResultsPage from './components/ResultsPage';

import BottomNavigationAction from '@material-ui/core/BottomNavigationAction';
import Typography from '@material-ui/core/Typography';
import {BottomNavigation } from "@material-ui/core";

import logo from './logo.svg';
import logoImg from './images/logoimgimproved.png';
import Email from '@material-ui/icons/Email';
import GithubIcon from './images/githubIcon.js';
import LinkedinIcon from './images/linkedinicon.js';

import Loader from './components/Loader'

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
        <Route path='/results' component={ResultsPage}/>
      </div>
      <div>
        <BottomNavigation className="bottomNav">
          <Typography variant="button" style={{ margin: 'auto' }}>Made by: Rafay Kalim</Typography>
          <BottomNavigationAction icon={<GithubIcon />} />
          <BottomNavigationAction icon={<Email />} />
          <BottomNavigationAction icon={<LinkedinIcon />} />
        </BottomNavigation>
      </div>
    </HashRouter>
  );
}

export default App;
