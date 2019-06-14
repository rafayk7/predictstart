import React, { Component } from 'react'
import { Navbar, NavItem, NavLink } from 'reactstrap'
import '../styles/App.css'

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
            <div>
                <h1>predicto</h1>
                <ul className="header">
                    <li><a href="/">Predict</a></li>
                    <div className="rightAlign">
                    <li><a href="/stuff">How it Works</a></li>
                    <li><a href="/contact">About Me</a></li>
                    </div>
                </ul>
                <div className="content">

                </div>
            </div>
        )
    }
}