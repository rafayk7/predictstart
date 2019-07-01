import React, { Component } from "react";
import { Redirect, withRouter } from 'react-router-dom';

import { TextField, Button } from "@material-ui/core";
import { MuiThemeProvider, createMuiTheme } from '@material-ui/core/styles';

import SvgIcon from 'material-ui/SvgIcon';
import lightGreen from '@material-ui/core/colors/lightGreen';
import { makeStyles } from '@material-ui/styles';

import scrapeIcon from '../images/scraping_tranparent.png';
import predictIcon from '../images/prediction.png';

import LoadingScreen from 'react-loading-screen';

import axios from 'axios';
import { apiurl } from '../utilities/LoadingUtility'

import Modal from 'react-modal';

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

const customStyles = {
    content : {
      top                   : '50%',
      left                  : '50%',
      right                 : 'auto',
      bottom                : 'auto',
      marginRight           : '-50%',
      transform             : 'translate(-50%, -50%)'
    }
  };

const HomeIcon = (props) => (
    <SvgIcon {...props}>
        <path d="M10 20v-6h4v6h5v-8h3L12 3 2 12h3v8z" />
    </SvgIcon>
);

class AddUrl extends Component {

    constructor(props) {
        super(props)

        this.state = {
            url: "",
            loading: false,
            loading_message: 'Scraping data...',
            icon: scrapeIcon,
            toResults: false,
            data: {},
            isModalOpen: false
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

        if(!this.state.url.startsWith('https://www.kickstarter.com/projects')){
            this.setState({isModalOpen: true})
        }

        else {
            this.setState({
                loading: true
            })
    
            axios.post(apiurl + '/predict', {
                url: this.state.url
            }).then((response) => {
                this.setState({
                    data: response.data,
                    loading: false,
                    loading_message: 'Scraping data...',
                    icon: scrapeIcon,
                    toResults: true
                })
                console.log(this.state.data)
            })
    
            setTimeout(() => {
                this.setState({
                    loading_message: 'Making prediction...',
                    icon: predictIcon
                })
                console.log("CHANGED")
            }, 1500)
    
            console.log("URL: " + this.state.url)
        }
    }


    render() {
        if (this.state.toResults) {
            // eslint-disable-next-line no-unused-expressions
            return <Redirect to={{
                pathname: '/results',
                state: this.state.data
            }} />
        }

        return (
            <React.Fragment>
                <Modal
                    isOpen={this.state.modalIsOpen}
                    onAfterOpen={this.afterOpenModal}
                    onRequestClose={this.closeModal}
                    style={customStyles}
                    contentLabel="Example Modal"
                >
                <h2>Hello</h2>

                </Modal>
                <LoadingScreen
                    loading={this.state.loading}
                    bgColor='#f1f1f1'
                    spinnerColor='#9ee5f8'
                    textColor='#676767'
                    logoSrc={this.state.icon}
                    text={this.state.loading_message}
                />

                <form onSubmit={this.handleSubmit.bind(this)}>
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
                                error={this.state.isModalOpen}
                                helperText={this.state.isModalOpen? 'Wrong Url! Make sure it starts with https://www.kickstarter.com/projects' : ' '}
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
                </form>
                <h2>predictstart is a website to make predictions on the success of any Kickstarter project.</h2>
                <h2>along with a prediction, it also gives you the top 5 features that contribute to the prediction made, to tell you what you should improve upon.</h2>
            </React.Fragment>
        );
    }



}

export default withRouter(AddUrl)