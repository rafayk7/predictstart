import React, { Component } from 'react';
import {withRouter} from 'react-router-dom'



class ResultsPage extends Component {

    constructor(props){
        super(props)
        console.log(this.props.location.state)

        if(this.props.location.state !== undefined){
            localStorage.setItem('data', JSON.stringify(this.props.location.state))
        }

        this.state = {
            data: JSON.parse(localStorage.getItem('data'))
        }
    }


    render() {
        return (
            <div>
                <span>
                    <h1>Prediction for {this.state.data.title}</h1>
                    <div className="login">

                    </div>
                </span>
            </div>


        )
    }

}

export default withRouter(ResultsPage)