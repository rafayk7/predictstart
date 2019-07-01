import React, { Component } from 'react';
import IconButton from '@material-ui/core/IconButton'
import Email from '@material-ui/icons/Email';
import github_icon from '../images/github_icon.png'
import { makeStyles } from '@material-ui/core/styles';


const useStyles = makeStyles({
    root: {
        width: 500,
    },
});

export default class AboutMe extends Component {
    render() {
        return (
            <div>
                <span>
                    <h1>
                        Me, Rafay
            </h1>
                    <p>I am a second year Engineering Science at the University of Toronto interested in Data Science, Machine Learning, Web/Mobile Dev </p>
                    <p>and creating end-to-end solutions that incorporate all of these aspects.</p>
                    <p>Feel free to reach out for any questions, or for anything else :)</p>
                </span>
            </div>


        )
    }

}