import React, { Component } from 'react'
import '../styles/App.css'
import {
    Route,
    HashRouter,
    NavLink
} from "react-router-dom";


export default class ToolBar extends Component {
    constructor(props) {
        super(props);

        this.toggle = this.toggle.bind(this);
        this.state = {
            isOpen: false
        };
    }
    toggle() {
        this.setState({
            isOpen: !this.state.isOpen
        });
    }

    render() {
        return (
            <HashRouter>
                <div>
                    <h1>predictstart</h1>
                    <h2>predictstart is a website to make predictions on the success of any Kickstarter project.</h2>
                    <h2>Along with a prediction, it also gives you the top 5 features that contribute to the prediction made, to tell you what you should improve upon.</h2>
                    <ul className="header">
                    <li><NavLink exact to="/">Predict</NavLink></li>

                        <div className="rightAlign">
                            <li><NavLink to="/howWorks">How it Works</NavLink></li>
                            <li><NavLink to="/aboutMe">About Me</NavLink></li>
                        </div>
                    </ul>
                    <div className="content">
                    </div>
                </div>
            </HashRouter>
        )
    }
}