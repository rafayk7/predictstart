import React, { Component } from 'react';
import { withRouter, Redirect } from 'react-router-dom'

import moment from 'moment'
import DataTable from './Datatable/Datatable';

import { List, ListItem } from 'material-ui/List';
import Subheader from 'material-ui/Subheader';
import Divider from 'material-ui/Divider';
import Checkbox from 'material-ui/Checkbox';
import Toggle from 'material-ui/Toggle';

import '../styles/App.css'

import MuiThemeProvider from 'material-ui/styles/MuiThemeProvider';

class ResultsPage extends Component {

    constructor(props) {
        super(props)
        console.log(this.props.location.state)

        if (this.props.location.state !== undefined) {
            localStorage.setItem('data', JSON.stringify(this.props.location.state))
        }

        var data = JSON.parse(localStorage.getItem('data'))
        var deadline = moment(data.deadline_date, "YYYY-MM-DD")

        var today = moment(new Date())
        var rows = []
        var headings = ['Feature', '% Contribution']
        var successSentence = ""

        if (!data.succeeded) {
            if (data.label[0] == 0) {
                successSentence = "Unfortunately I don't think this project will succeed :(. Please look at the features that affect this the most."
            } else if (data.label[0] == 1) {
                successSentence = "Amazing! The prospects for this project look good :). Please look at the features that affect this the most."
            }

            var shap_vals = data.imp_vals[0]
            var sum = 0
            var n = 0

            //If imp_vals exist, calculate % contrib of each of important feature
            if (data.imp_vals) {
                for (var key in shap_vals) {
                    n = n + 1
                    sum = sum + shap_vals[key]
                }

                for (var key in shap_vals) {
                    shap_vals[key] = shap_vals[key] / sum * 100
                }

                //Sort dictionary to get 
                var feature_imps = Object.keys(shap_vals).map(function (key) {
                    return [key, shap_vals[key]];
                });

                feature_imps.sort(function (first, second) {
                    return second[1] - first[1];
                });

                console.log(feature_imps)
                rows = feature_imps
            }
        }

        this.state = {
            data: data,
            days_remaining: Math.round(moment.duration(deadline.diff(today)).asDays()),
            headings: headings,
            rows: rows.slice(0,5),
            successSentence: successSentence,
            error: data.error
        }
    }

    
    render() {
        if (this.state.data.succeeded) {
            return (
                <div>
                    <span>
                        <h1>Prediction for {this.state.data.title}</h1>
                        <h2>This project already succeeded! ${this.state.data.usd_pledged} pledged from a ${this.state.data.goal} goal</h2>
                        <h2>Launched on {this.state.data.launched_date}, and a deadline of {this.state.data.deadline_date}, you still have {this.state.days_remaining} days remaining.</h2>
                        <h2>Congratulations! :)</h2>
                    </span>
                </div>
            )
        }
        if (this.state.error) {
            return <Redirect to='/' />
        }
        return (
            <div>
                <span>
                    <h1>Prediction for {this.state.data.title}</h1>
                    <h2>This project has not succeeded yet. Only ${this.state.data.goal - this.state.data.usd_pledged} left from a ${this.state.data.goal} goal!</h2>
                    <h2>{this.state.successSentence}</h2>
                    <div>
                    </div>
                    {/* <DataTable headings = {this.state.headings} rows ={this.state.rows}></DataTable> */}
                    <DataTable headings={this.state.headings} rows={this.state.rows}></DataTable>
                </span>
            </div >
        )
    }

}

export default withRouter(ResultsPage)