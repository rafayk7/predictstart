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