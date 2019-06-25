import React, { Component } from "react";
import { TextField, Button, Toolbar, BottomNavigation } from "@material-ui/core";
import { MuiThemeProvider, createMuiTheme } from '@material-ui/core/styles';
import Typography from '@material-ui/core/Typography';

import BottomNavigationAction from '@material-ui/core/BottomNavigationAction';
import SvgIcon from 'material-ui/SvgIcon';
import Icon from '@material-ui/core/Icon';

import Email from '@material-ui/icons/Email';
import lightGreen from '@material-ui/core/colors/lightGreen';
import GithubIcon from '../images/githubIcon.js';
import LinkedinIcon from '../images/linkedinicon.js';

import { makeStyles } from '@material-ui/styles';

// import githubIcon from '../images/github_icon.svg'
import '../styles/App.css';

const blackTheme = createMuiTheme({ palette: { primary: lightGreen } })

const useStyles = makeStyles({
    imageIcon: {
        height: '100%'
    },
    iconRoot: {
        textAlign: 'center'
    }
});

const HomeIcon = (props) => (
    <SvgIcon {...props}>
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
    </SvgIcon>
);

export default class AddUrl extends Component {

    constructor(props) {
        super(props)

        this.state = {
            url: "",

        };

        const styles = {
            button: {
                width: '150px'
            },
            urlField: {
                width: '300px'
            }
        }


    }

    // changeHandler = (event) => {
    //     const url = event.target.value;
    //     this.setState({
    //         formControls: {
    //             url: url,
    //         }
    //     })
    //     console.log("entered: " + url)
    //     console.log(this.state.formControls.url);
    // }

    validateForm() {
        return this.state.url.length
    }

    handleChange = event => {
        this.setState({
            [event.target.id]: event.target.value
        });
        console.log(this.state[event.target.id])
    }

    handleSubmit = event => {
        event.preventDefault();
        console.log("URL: " + this.state.url)
    }


    render() {
        return (
            <form onSubmit={this.handleSubmit}>
                <span>
                    <h1>Enter your project URL here.</h1>
                    <MuiThemeProvider theme={blackTheme}>

                        <TextField
                            id="url"
                            label="URL"
                            value={this.state.url}
                            className="input"
                            onChange={this.handleChange}
                            margin="normal"
                            variant="outlined"
                        />
                    </MuiThemeProvider>

                </span>
                <div>
                    <span>
                        <MuiThemeProvider theme={blackTheme}>

                            <Button
                                // className="button"
                                disabled={!this.validateForm()}
                                type="submit"
                                variant="outlined"
                                color="primary"
                            >
                                Predict
          </Button>
                        </MuiThemeProvider>

                    </span>
                </div>
                <BottomNavigation className="bottomNav">
                    <Typography variant="button" style={{margin: 'auto'}}>Made by: Rafay Kalim</Typography>
                    <BottomNavigationAction icon={<GithubIcon />} />
                    <BottomNavigationAction icon={<Email />} />
                    <BottomNavigationAction icon={<LinkedinIcon/>} />
                </BottomNavigation>
            </form>


        );
    }



}