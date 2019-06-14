import React, { Component } from "react";
import { TextField, Button, Toolbar } from "@material-ui/core"
import '../styles/App.css';

export default class AddUrl extends Component {
    constructor(props) {
        super(props)

        this.state = {
            url: "",
        };


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
                <div className="login">
                    <span>
                        <h1>Enter your project URL here.</h1>
                        <TextField
                            id="url"
                            label="url"
                            value={this.state.url}
                            className="input"
                            onChange={this.handleChange}
                            margin="normal"
                            variant="outlined"
                            color=""
                        />
                    </span>
                </div>
                <div className="login">
                    <span>
                        <Button
                            className="button"
                            disabled={!this.validateForm()}
                            type="submit"
                            variant="outlined"
                            color="primary"
                        >
                            Predict
          </Button>
                    </span>
                </div>
            </form>
        );
    }



}